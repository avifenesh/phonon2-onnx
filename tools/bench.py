"""CPU timing and RAM for one variant, one process launch (run it N times from the shell and take medians).

Pipeline timed end to end: preprocessor-model.onnx -> encoder -> greedy TDT loop over decoder_joint-model.onnx (the
reference loop in parity.py), B = 1, ORT CPU EP, intra_op_num_threads = --threads, inter_op 1, default graph
optimisation. Prints one JSON line: load time, RSS after load, peak RSS, and per clip the median of --runs timed runs
after one warm-up (preprocess, encoder, decode, total).

Usage: python tools/bench.py --suffix "" | exact4x2 | int8  [--threads 4] [--runs 3]
       python tools/bench.py --fermion default|fp32   (Fermion's own runtime on the same clips, for context)
"""
import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
CLIPS = ["ls_1272-141231-0006", "ls_1272-128104-0004", "own_1790825456375"]


def rss_mb():
    with open("/proc/self/status") as fh:
        for l in fh:
            if l.startswith("VmRSS:"):
                return int(l.split()[1]) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--fermion", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    man = {j["id"]: j for j in map(json.loads, open(ROOT / "work" / "manifest.jsonl"))}
    wavs = {c: sf.read(man[c]["path"], dtype="float32")[0] for c in CLIPS}
    r0 = rss_mb(); t0 = time.perf_counter()
    if a.fermion:
        os.environ["FERMION_CPU_THREADS"] = str(a.threads)
        if a.fermion == "fp32":
            os.environ["FERMION_P2_CPU"] = "fp32"; os.environ["FERMION_P2_CPU_TDT"] = "torch"
        import torch
        torch.set_num_threads(a.threads)
        from fermion._speech import engine_phonon2_cpu as e2
        sp = e2.load(Path.home() / ".cache/fermion/speech/FermionResearch__Phonon-2/model_phonon2_c4c_int6", profile="phonon-2", backend="cpu", quiet=True)

        def run(w):
            t = time.perf_counter(); sp._decode_single(w, 1.05); return {"total": time.perf_counter() - t}
        tag = f"fermion-{a.fermion}"
    else:
        import onnxruntime as rt
        from parity import ref_decode
        dist = ROOT / "dist"; sfx = f".{a.suffix}" if a.suffix else ""
        o = rt.SessionOptions(); o.intra_op_num_threads = a.threads; o.inter_op_num_threads = 1
        P = ["CPUExecutionProvider"]
        pre = rt.InferenceSession(str(dist / "preprocessor-model.onnx"), o, providers=P)
        enc = rt.InferenceSession(str(dist / f"encoder-model{sfx}.onnx"), o, providers=P)
        dj = rt.InferenceSession(str(dist / "decoder_joint-model.onnx"), o, providers=P)

        def run(w):
            t = time.perf_counter()
            f, fl = pre.run(None, {"waveforms": w[None], "waveforms_lens": np.array([len(w)], np.int64)}); t1 = time.perf_counter()
            e, el = enc.run(None, {"audio_signal": f, "length": fl}); t2 = time.perf_counter()
            ref_decode(dj, e[0], int(el[0])); t3 = time.perf_counter()
            return {"pre": t1 - t, "enc": t2 - t1, "dec": t3 - t2, "total": t3 - t}
        tag = "onnx-" + (a.suffix or "fp32")
    load_s = time.perf_counter() - t0; r_load = rss_mb()
    out = {"variant": tag, "threads": a.threads, "load_s": round(load_s, 2), "rss_after_load_mb": round(r_load), "clips": {}}
    for c, w in wavs.items():
        run(w)
        rs = [run(w) for _ in range(a.runs)]
        out["clips"][c] = {"audio_s": round(len(w) / 16000, 2), **{k: round(statistics.median(x[k] for x in rs), 4) for k in rs[0]}}
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    out["rss_before_load_mb"] = round(r0)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
