"""Export Phonon-2 (Fermion five-value container) to ONNX in the onnx-asr `nemo-conformer-tdt` layout.

Weights come from Fermion's own loader (`fermion._speech.engine_phonon2_cpu.load` with FERMION_P2_CPU=fp32), which
expands every five-value linear exactly to dense fp32 and the int6/fp16 tables to fp32 inside the stock transformers
ParakeetForTDT graph. The ONNX graphs are traced from that same torch model, so no weight is re-derived here.

Writes to dist/ (or --out):
  preprocessor-model.onnx   waveforms -> log-mel features, Fermion's front end (see README for why not nemo128)
  encoder-model.onnx(+.data) features -> encoder states (1024-d, before the joint's encoder projection)
  decoder_joint-model.onnx  one TDT step: prediction net (embedding + 2-layer LSTM + projection) and joint
  vocab.txt, config.json    onnx-asr metadata
"""
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("PHONON2_MODEL_DIR", Path.home() / ".cache/fermion/speech/FermionResearch__Phonon-2/model_phonon2_c4c_int6"))
OPSET = 17


def load_fp32_model(threads):
    os.environ["FERMION_P2_CPU"] = "fp32"
    os.environ["FERMION_P2_CPU_TDT"] = "torch"
    os.environ["FERMION_CPU_THREADS"] = str(threads)
    import torch
    torch.set_num_threads(threads)
    from fermion._speech import engine_phonon2_cpu as e2
    sp = e2.load(MODEL_DIR, profile="phonon-2", backend="cpu")
    assert sp._cenc is None and sp._ctdt is None and not sp._bf16
    return sp


def build_modules(sp):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from fermion._speech.engine_phonon2_cpu import N_FFT, HOP, WIN, N_MELS, PREEMPH, LOG_GUARD

    class Preprocessor(nn.Module):
        """Fermion's `_log_mel`, batched. Frames = len // 160 + 1 (torch.stft center=True, zero pad), all of them valid
        and all of them in the per-feature mean/std (unbiased). Samples past waveforms_lens are zeroed after pre-emphasis,
        so a padded batch row gives the same features as the row alone."""

        def __init__(self):
            super().__init__()
            win = torch.hann_window(WIN, periodic=False, dtype=torch.float64)
            lp = (N_FFT - WIN) // 2
            win = F.pad(win, (lp, N_FFT - WIN - lp))
            n = torch.arange(N_FFT, dtype=torch.float64); k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)
            ang = 2 * math.pi * k[:, None] * n[None, :] / N_FFT
            w = torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win], 0).to(torch.float32)[:, None, :]
            self.register_buffer("dft", w)                                        # [514, 1, 512]
            self.register_buffer("melf", sp._melf.clone().to(torch.float32))      # [128, 257] slaney
            self.nb = N_FFT // 2 + 1

        def forward(self, waveforms, waveforms_lens):
            x = torch.cat([waveforms[:, :1], waveforms[:, 1:] - PREEMPH * waveforms[:, :-1]], dim=1)
            idx = torch.arange(x.shape[1], device=x.device)[None, :]
            x = torch.where(idx < waveforms_lens[:, None], x, torch.zeros_like(x))
            x = F.pad(x, (N_FFT // 2, N_FFT // 2))
            spec = F.conv1d(x[:, None, :], self.dft, stride=HOP)                   # [B, 514, F]
            re, im = spec[:, : self.nb], spec[:, self.nb:]
            power = torch.sqrt(re * re + im * im).pow(2)
            mel = torch.log(torch.matmul(self.melf, power) + LOG_GUARD)            # [B, 128, F]
            flen = waveforms_lens // HOP + 1
            fidx = torch.arange(mel.shape[2], device=x.device)[None, None, :]
            mask = (fidx < flen[:, None, None]).to(mel.dtype)
            n = flen.to(mel.dtype)[:, None, None]
            mean = (mel * mask).sum(2, keepdim=True) / n
            var = (((mel - mean) * mask) ** 2).sum(2, keepdim=True) / (n - 1)
            feats = (mel - mean) / (torch.sqrt(var) + 1e-5) * mask
            return feats, flen

    m = sp.model

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = m.encoder

        def forward(self, audio_signal, length):
            feats = audio_signal.transpose(1, 2)                                   # [B, T, 128]
            mask = (torch.arange(feats.shape[1], device=feats.device)[None, :] < length[:, None]).to(torch.long)
            out = self.enc(input_features=feats, attention_mask=mask).last_hidden_state
            lengths = length.to(torch.float32)
            for _ in range(3):                                                     # ParakeetPreTrainedModel._get_subsampling_output_length
                lengths = torch.floor((lengths - 1.0) / 2.0 + 1.0)
            return out.transpose(1, 2), lengths.to(torch.int64)

    class DecoderJoint(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = m.decoder.embedding; self.lstm = m.decoder.lstm; self.pred = m.decoder.decoder_projector
            self.encp = m.encoder_projector; self.act = m.joint.activation; self.head = m.joint.head

        def forward(self, encoder_outputs, targets, target_length, input_states_1, input_states_2):
            e = self.emb(targets.to(torch.int64))                                   # [B, U, 640]
            out, (h, c) = self.lstm(e, (input_states_1, input_states_2))
            dec = self.pred(out)                                                   # [B, U, 640]
            enc = self.encp(encoder_outputs.transpose(1, 2))                       # [B, T, 640]
            logits = self.head(self.act(enc[:, :, None, :] + dec[:, None, :, :]))  # [B, T, U, 8198]
            return logits, target_length, h, c

    return Preprocessor().eval(), Encoder().eval(), DecoderJoint().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default="pre,enc,dec,meta")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(","))
    sp = load_fp32_model(a.threads)
    import torch
    import onnx
    for mod in (sp.model, sp.model.encoder):
        mod.config._attn_implementation = "eager"
    pre, enc, dj = build_modules(sp)

    with torch.inference_mode():
        if "pre" in only:
            wav = torch.randn(2, 48000) * 0.1; lens = torch.tensor([48000, 40000])
            torch.onnx.export(pre, (wav, lens), str(out / "preprocessor-model.onnx"), dynamo=False, opset_version=OPSET,
                              input_names=["waveforms", "waveforms_lens"], output_names=["features", "features_lens"],
                              dynamic_axes={"waveforms": {0: "batch", 1: "samples"}, "waveforms_lens": {0: "batch"},
                                            "features": {0: "batch", 2: "frames"}, "features_lens": {0: "batch"}})
            print("preprocessor exported", file=sys.stderr)
        if "dec" in only:
            ex = (torch.randn(1, 1024, 1), torch.tensor([[8192]], dtype=torch.int32), torch.tensor([1], dtype=torch.int32),
                  torch.zeros(2, 1, 640), torch.zeros(2, 1, 640))
            torch.onnx.export(dj, ex, str(out / "decoder_joint-model.onnx"), dynamo=False, opset_version=OPSET,
                              input_names=["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"],
                              output_names=["outputs", "prednet_lengths", "output_states_1", "output_states_2"],
                              dynamic_axes={"encoder_outputs": {0: "batch", 2: "enc_time"}, "targets": {0: "batch", 1: "tgt_len"},
                                            "target_length": {0: "batch"}, "input_states_1": {1: "batch"}, "input_states_2": {1: "batch"},
                                            "outputs": {0: "batch", 1: "enc_time", 2: "tgt_len"}, "prednet_lengths": {0: "batch"},
                                            "output_states_1": {1: "batch"}, "output_states_2": {1: "batch"}})
            print("decoder_joint exported", file=sys.stderr)
        if "enc" in only:
            feats = torch.randn(1, 128, 501); lens = torch.tensor([501])
            tmpdir = out / "_enc_tmp"; tmpdir.mkdir(exist_ok=True)
            tmp = tmpdir / "enc.onnx"                          # >2 GB: torch writes one external file per tensor here
            torch.onnx.export(enc, (feats, lens), str(tmp), dynamo=False, opset_version=OPSET,
                              input_names=["audio_signal", "length"], output_names=["outputs", "encoded_lengths"],
                              dynamic_axes={"audio_signal": {0: "batch", 2: "time"}, "length": {0: "batch"},
                                            "outputs": {0: "batch", 2: "enc_time"}, "encoded_lengths": {0: "batch"}})
            mp = onnx.load(str(tmp))
            onnx.save(mp, str(out / "encoder-model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                      location="encoder-model.onnx.data", size_threshold=1024)
            del mp
            shutil.rmtree(tmpdir)
            print("encoder exported", file=sys.stderr)
    if "meta" in only:
        vocab = list(sp._vocab) + ["<blk>"]
        assert len(vocab) == 8193 and vocab[8192] == "<blk>"
        with open(out / "vocab.txt", "w", encoding="utf-8") as fh:
            for i, t in enumerate(vocab):
                fh.write(f"{t} {i}\n")
        cfg = {"model_type": "nemo-conformer-tdt", "features_size": 128, "subsampling_factor": 8, "max_tokens_per_step": 10}
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    main()
