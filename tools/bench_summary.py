"""Median and min..max across process launches of tools/bench.py output (work/bench/*.jsonl)."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = {}
for f in sorted((ROOT / "work" / "bench").glob("*.jsonl")):
    rs = [json.loads(l) for l in open(f) if l.strip()]
    s = {"launches": len(rs)}
    for k in ("load_s", "rss_after_load_mb", "peak_rss_mb"):
        v = [r[k] for r in rs]; s[k] = [statistics.median(v), min(v), max(v)]
    for c in rs[0]["clips"]:
        s[c] = {"audio_s": rs[0]["clips"][c]["audio_s"]}
        for k in rs[0]["clips"][c]:
            if k == "audio_s":
                continue
            v = [r["clips"][c][k] for r in rs]; s[c][k] = [round(statistics.median(v), 3), round(min(v), 3), round(max(v), 3)]
        s[c]["rtf_x"] = round(s[c]["audio_s"] / s[c]["total"][0], 1)
    out[f.stem] = s
(ROOT / "work" / "bench_summary.json").write_text(json.dumps(out, indent=1))
for k, s in out.items():
    print(k, json.dumps(s))
