"""WER on the LibriSpeech clips (reference text in the manifest) and transcript disagreement on every clip.

Normalisation for WER: uppercase, '-' -> space, drop everything except A-Z, 0-9, apostrophe and space.
Usage: python tools/wer.py  (reads work/ref/{fp32,default}.jsonl and work/parity/*.jsonl)
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(s):
    s = s.upper().replace("-", " ")
    return re.sub(r"[^A-Z0-9' ]", "", s).split()


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def main():
    man = {j["id"]: j for j in map(json.loads, open(ROOT / "work" / "manifest.jsonl"))}
    systems = {"fermion-fp32": ROOT / "work/ref/fp32.jsonl", "fermion-default": ROOT / "work/ref/default.jsonl"}
    for f in sorted((ROOT / "work" / "parity").glob("*.jsonl")):
        systems[f"onnx-{f.stem}"] = f
    hyp = {k: {j["id"]: j["text"] for j in map(json.loads, open(p))} for k, p in systems.items()}
    base = hyp["fermion-fp32"]
    out = {}
    for k, h in hyp.items():
        e = n = 0
        for cid, r in man.items():
            if r["ref"] is None:
                continue
            ref = norm(r["ref"]); e += edits(ref, norm(h[cid])); n += len(ref)
        de = dn = 0
        for cid in man:
            b = norm(base[cid]); de += edits(b, norm(h[cid])); dn += len(b)
        out[k] = {"librispeech_wer_pct": round(100 * e / n, 3), "librispeech_words": n,
                  "word_diff_vs_fermion_fp32_pct_all_clips": round(100 * de / dn, 3),
                  "texts_identical_to_fermion_fp32": sum(h[c] == base[c] for c in man), "clips": len(man)}
    (ROOT / "work" / "wer.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:24s} {v}")


if __name__ == "__main__":
    main()
