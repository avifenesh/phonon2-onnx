"""Parity of the ONNX export against Fermion's runtime outputs recorded by tools/fermion_ref.py.

For every clip in work/manifest.jsonl:
  * onnx-asr path: onnx_asr's NemoConformerTdt class loads dist/ (encoder, decoder_joint, vocab, config) and does its
    own greedy TDT decode. The only override is the preprocessor: onnx-asr's built-in nemo128 is swapped for
    dist/preprocessor-model.onnx (Fermion's front end). `--stock-pre` keeps onnx-asr's nemo128 instead.
  * reference loop: a plain-Python greedy TDT loop over the same ONNX sessions, written the way the Rust daemon is
    meant to implement it (see README "Decoding").
Token ids and text are compared with Fermion fp32 (exact-weights reference) and Fermion default (packed int8 path).
With --dump-diff, features and encoder outputs are compared numerically on the clips fermion_ref.py dumped.

Usage: python tools/parity.py [--dist dist] [--suffix int8] [--stock-pre] [--tag NAME]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
BLANK, VOCAB_N, DURATIONS, MAX_SYMBOLS = 8192, 8193, (0, 1, 2, 3, 4), 10


def special(piece):
    return (piece.startswith("<|") and piece.endswith("|>")) or piece in ("<unk>", "<pad>")


def detok(ids, vocab):
    """Fermion's rule: drop special pieces, join, '▁' -> ' ', strip."""
    return "".join(vocab[i] for i in ids if not special(vocab[i])).replace("▁", " ").strip()


def so(threads):
    o = rt.SessionOptions(); o.intra_op_num_threads = threads; o.inter_op_num_threads = 1
    return o


def ref_decode(dj, enc, T):
    """enc: [1024, T_enc] float32 for one utterance. Greedy TDT, Fermion's loop."""
    h = np.zeros((2, 1, 640), np.float32); c = np.zeros((2, 1, 640), np.float32)
    t, last, ids, nsym = 0, BLANK, [], 0
    while t < T:
        logits, _, h2, c2 = dj.run(None, {"encoder_outputs": enc[None, :, t:t + 1], "targets": np.array([[last]], np.int32),
                                          "target_length": np.array([1], np.int32), "input_states_1": h, "input_states_2": c})
        lg = logits[0, 0, 0]
        tok = int(lg[:VOCAB_N].argmax()); dur = DURATIONS[int(lg[VOCAB_N:].argmax())]
        if tok == BLANK and dur == 0:
            dur = 1
        if tok != BLANK:
            ids.append(tok); last = tok; h, c = h2, c2
        if dur == 0:
            nsym += 1
            if nsym >= MAX_SYMBOLS:
                dur = 1; nsym = 0
        else:
            nsym = 0
        t += dur
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default=str(ROOT / "dist"))
    ap.add_argument("--suffix", default="", help="onnx-asr quantization suffix, e.g. int8 -> encoder-model.int8.onnx")
    ap.add_argument("--stock-pre", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-ref-loop", action="store_true")
    a = ap.parse_args()
    dist = Path(a.dist)
    tag = a.tag or ("fp32" if not a.suffix else a.suffix) + ("-stockpre" if a.stock_pre else "")

    import onnx_asr
    from onnx_asr.loader import Manager
    from onnx_asr.preprocessors.preprocessor import OnnxPreprocessor

    class PhononManager(Manager):
        def _create_preprocessor(self, name):
            if a.stock_pre:
                return super()._create_preprocessor(name)
            assert name == "nemo128", name
            p = OnnxPreprocessor.__new__(OnnxPreprocessor)
            p._preprocessor = rt.InferenceSession(str(dist / "preprocessor-model.onnx"), so(a.threads), providers=["CPUExecutionProvider"])
            return p

    mgr = PhononManager(sess_options=so(a.threads), providers=["CPUExecutionProvider"])
    model = mgr.create_asr("nemo-conformer-tdt", dist, quantization=a.suffix or None)
    asr = model.asr
    vocab = [l.rstrip("\n").rsplit(" ", 1)[0] for l in open(dist / "vocab.txt", encoding="utf-8")]

    pre = rt.InferenceSession(str(dist / "preprocessor-model.onnx"), so(a.threads), providers=["CPUExecutionProvider"])
    sfx = f".{a.suffix}" if a.suffix else ""
    encs = asr._encoder; dj = asr._decoder_joint

    refs = {m: {j["id"]: j for j in map(json.loads, open(ROOT / "work" / "ref" / f"{m}.jsonl"))} for m in ("fp32", "default")}
    rows = [json.loads(l) for l in open(ROOT / "work" / "manifest.jsonl")]
    res = []
    for r in rows:
        w, sr = sf.read(r["path"], dtype="float32"); assert sr == 16000
        t0 = time.perf_counter()
        text_asr = model.recognize(w, sample_rate=16000)
        dt = time.perf_counter() - t0
        wl = np.array([len(w)], np.int64)
        feats, flen = asr._preprocessor(w[None], wl)
        eo, el = asr._encode(feats, flen)
        ids_asr = [int(x) for x in next(iter(asr._decoding(eo, el)))[0]]
        ids_ref = None
        if not a.no_ref_loop:
            ids_ref = ref_decode(dj, eo[0].T.copy(), int(el[0]))
            assert ids_ref == ids_asr, (r["id"], "reference loop != onnx-asr loop")
        x = {"id": r["id"], "ids": ids_asr, "text_onnx_asr": text_asr, "text": detok(ids_asr, vocab), "seconds": round(dt, 4)}
        for m in ("fp32", "default"):
            f = refs[m][r["id"]]
            x[f"ids_eq_{m}"] = ids_asr == f["ids"]; x[f"text_eq_{m}"] = x["text"] == f["text"]
        res.append(x)
        flag = "" if x["ids_eq_fp32"] else "  <-- differs from fermion fp32"
        print(f"{r['id']}  {dt:.2f}s  {x['text'][:70]}{flag}", file=sys.stderr)

    out = ROOT / "work" / "parity" / f"{tag}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for x in res:
            fh.write(json.dumps(x) + "\n")
    n = len(res)
    summ = {"tag": tag, "clips": n}
    for m in ("fp32", "default"):
        summ[f"ids_eq_{m}"] = sum(x[f"ids_eq_{m}"] for x in res); summ[f"text_eq_{m}"] = sum(x[f"text_eq_{m}"] for x in res)
    summ["onnx_asr_text_eq_fermion_rule"] = sum(x["text_onnx_asr"] == x["text"] for x in res)
    summ["onnx_asr_text_eq_fermion_fp32"] = sum(x["text_onnx_asr"] == refs["fp32"][x["id"]]["text"] for x in res)

    # numerics on the dumped clips
    diffs = {}
    for f in sorted((ROOT / "work" / "ref" / "fp32").glob("*.npz")):
        cid = f.stem; r = next(rr for rr in rows if rr["id"] == cid)
        w, _ = sf.read(r["path"], dtype="float32")
        feats, flen = pre.run(None, {"waveforms": w[None], "waveforms_lens": np.array([len(w)], np.int64)})
        eo, el = encs.run(None, {"audio_signal": feats, "length": flen})
        d = {}
        for m in ("fp32", "default"):
            rf = np.load(ROOT / "work" / "ref" / m / f"{cid}.npz")
            F, E = rf["feats"], rf["enc"]
            assert F.shape == feats[0].T.shape and E.shape == eo[0].T.shape, (F.shape, feats.shape, E.shape, eo.shape)
            d[m] = {"feat_max_abs": float(np.abs(F - feats[0].T).max()), "enc_max_abs": float(np.abs(E - eo[0].T).max()),
                    "enc_ref_max_abs": float(np.abs(E).max()),
                    "enc_rel_l2": float(np.linalg.norm(E - eo[0].T) / np.linalg.norm(E))}
            # encoder alone: Fermion's own features into the ONNX encoder
            e2, _ = encs.run(None, {"audio_signal": np.ascontiguousarray(F.T[None]), "length": np.array([F.shape[0]], np.int64)})
            d[m]["enc_from_ref_feats_max_abs"] = float(np.abs(E - e2[0].T).max())
        d["frames"] = int(flen[0]); d["enc_frames"] = int(el[0])
        diffs[cid] = d
    summ["numerics"] = diffs
    (ROOT / "work" / "parity" / f"{tag}.summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
