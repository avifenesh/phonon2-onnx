"""Run Fermion's own Phonon-2 CPU runtime over the manifest and record its outputs.

Two modes of the same `fermion._speech.engine_phonon2_cpu` engine:
  fp32     FERMION_P2_CPU=fp32, FERMION_P2_CPU_TDT=torch: the five-value container expanded exactly to dense fp32 in the
           stock transformers ParakeetForTDT graph, torch greedy TDT loop. This is the exact-weights reference.
  default  what `fermion transcribe phonon-2` runs on this machine: packed one-dot int8 GEMMs (int8 activations),
           C encoder, C TDT loop (int6 decoder tables, int8 activations).

Per clip it records the transcript text the runtime returns and the token ids. In default mode the ids are taken from
the C TDT loop's own return value. In fp32 mode they come from a verbatim copy of the engine's torch loop, asserted to
give the same text as the engine. With --dump, features and encoder outputs (1024-d, before encoder_projector) are
written to work/ref/<mode>/<id>.npz.

Usage: python tools/fermion_ref.py {fp32|default} [--dump ID,ID]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("PHONON2_MODEL_DIR", Path.home() / ".cache/fermion/speech/FermionResearch__Phonon-2/model_phonon2_c4c_int6"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fp32", "default"])
    ap.add_argument("--dump", default="")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.environ["FERMION_CPU_THREADS"] = str(a.threads)
    if a.mode == "fp32":
        os.environ["FERMION_P2_CPU"] = "fp32"
        os.environ["FERMION_P2_CPU_TDT"] = "torch"
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from fermion._speech import engine_phonon2_cpu as e2
    from fermion._speech.engine import read_audio

    sp = e2.load(MODEL_DIR, profile="phonon-2", backend="cpu")
    print(json.dumps(sp.describe()["runtime"]), file=sys.stderr)
    if a.mode == "default":
        assert sp._cenc is not None and sp._ctdt is not None, "default mode expects the C encoder and C TDT loop"
    else:
        assert sp._cenc is None and sp._ctdt is None and not sp._bf16

    dump = set(x for x in a.dump.split(",") if x)
    outdir = ROOT / "work" / "ref" / a.mode
    outdir.mkdir(parents=True, exist_ok=True)
    cap = {}

    if a.mode == "default":
        cenc, ctdt = sp._cenc, sp._ctdt
        f_enc, f_tdt = cenc.forward, ctdt.decode

        def enc_fwd(feats):
            cap["feats"] = np.array(feats); out = f_enc(feats); cap["enc"] = np.array(out); return out

        def tdt_dec(encp):
            ids = f_tdt(encp); cap["ids"] = list(ids); return ids
        cenc.forward = enc_fwd; ctdt.decode = tdt_dec
    else:
        m = sp.model

        def hook(mod, inp, out):
            cap["enc"] = out.last_hidden_state[0].detach().numpy().copy()
        m.encoder.register_forward_hook(hook)

    def fp32_ids(audio):
        """Verbatim copy of Phonon2CpuSpeechModel._decode_single's torch loop (fp32, no autocast), returning ids."""
        m = sp.model; cfg = m.config; blank, V = cfg.blank_token_id, cfg.vocab_size
        with torch.inference_mode():
            feats, am = sp._log_mel(audio)
            enc = m.encoder(input_features=feats, attention_mask=am).last_hidden_state
            T = int(m._get_subsampling_output_length(am.sum(-1))[0])
            encp = m.encoder_projector(enc)[0]
            t = 0; last = blank; ids = []; nsym = 0; it = 0
            h = torch.zeros(m.decoder.lstm.num_layers, 1, m.decoder.lstm.hidden_size); c = torch.zeros_like(h)
            while t < T and it < sp._max_sym * T + 16:
                emb = m.decoder.embedding(torch.tensor([[last]]))
                out, (h2, c2) = m.decoder.lstm(emb, (h, c)); dec = m.decoder.decoder_projector(out[0])
                logits = m.joint.head(m.joint.activation(encp[t][None] + dec))[0]
                tok = int(logits[:V].argmax()); dur = int(sp._durations[int(logits[V:].argmax())]); it += 1
                if tok == blank and dur == 0:
                    dur = 1
                if tok != blank:
                    ids.append(tok); last = tok; h, c = h2, c2
                if dur == 0:
                    nsym += 1
                    if nsym >= sp._max_sym:
                        dur = 1; nsym = 0
                else:
                    nsym = 0
                t += dur
        return ids, feats[0].numpy().copy(), T

    rows = [json.loads(l) for l in open(ROOT / "work" / "manifest.jsonl")]
    results = []
    t_all = time.perf_counter()
    for r in rows:
        audio = read_audio(r["path"])
        assert len(audio) / 16000 <= 35.0, "single-shot regime only"
        cap.clear()
        t0 = time.perf_counter()
        text, _, _ = sp._decode_single(audio, 1.05)
        dt = time.perf_counter() - t0
        enc = cap.get("enc")
        if a.mode == "fp32":
            ids, feats, T = fp32_ids(audio)
            vocab = sp._vocab
            t2 = "".join(vocab[i] for i in ids if i < len(vocab) and not e2._special(vocab[i])).replace("▁", " ").strip()
            assert t2 == text, (r["id"], t2, text)
        else:
            ids, feats = cap["ids"], cap["feats"]
        results.append({"id": r["id"], "text": text, "ids": ids, "seconds": round(dt, 4), "audio_s": round(len(audio) / 16000, 3)})
        if r["id"] in dump:
            np.savez(outdir / f"{r['id']}.npz", feats=feats, enc=enc)
        print(f"{r['id']}  {dt:.2f}s  {text[:80]}", file=sys.stderr)
    with open(ROOT / "work" / "ref" / f"{a.mode}.jsonl", "w") as fh:
        for x in results:
            fh.write(json.dumps(x) + "\n")
    print(f"{len(results)} clips in {time.perf_counter() - t_all:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
