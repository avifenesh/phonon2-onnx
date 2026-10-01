"""Census of the five-value encoder linears in model.fermion (read with Fermion's own container reader).

Per tensor: per-row level set {-hi, -lo, 0, lo, hi} (fp16 lo/hi), occupancy of each level, and whether the level set is a
uniform integer grid: hi/lo == b/a exactly with 1 <= a < b <= 7 (what a single 4-bit weight grid, q - 8 in [-8, 7],
could hold exactly) or b <= 127 (8-bit). Writes work/weights_census.json and prints a summary.
"""
import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("PHONON2_MODEL_DIR", Path.home() / ".cache/fermion/speech/FermionResearch__Phonon-2/model_phonon2_c4c_int6"))


def main():
    from fermion._speech._engine_phonon2 import load
    fc = load("fermion_container")
    rows_total = 0; uni4 = 0; uni8 = 0; two = 0; eq = 0; zero_rows = 0
    ratios = []; occ = np.zeros(5); per = []; n_fv = 0; params = 0
    with open(MODEL_DIR / "model.fermion", "rb") as fh:
        n = int.from_bytes(fh.read(8), "little"); header = json.loads(fh.read(n))
        kinds = {}
        for e in header["index"]:
            blob = fh.read(e["b"]); kinds[e["k"]] = kinds.get(e["k"], 0) + 1
            if e["k"] != "five_value":
                continue
            n_fv += 1
            r = {}; w = fc._five_value(blob, tuple(e["shape"]), r)
            o, i = e["shape"]; params += o * i
            lo = r["lo"].astype(np.float64); hi = r["hi"].astype(np.float64)
            sign = r["sign"]; ish = r["is_hi"]
            # the dense expansion is exactly sign * (is_hi ? hi : lo)
            assert np.array_equal(w.astype(np.float32), (sign * np.where(ish, hi[:, None], lo[:, None])).astype(np.float16).astype(np.float32))
            used_hi = (ish & (sign != 0)).any(1); used_lo = ((~ish) & (sign != 0)).any(1)
            zero_rows += int((~(used_hi | used_lo)).sum())
            for k, v in enumerate((-2, -1, 0, 1, 2)):
                if v == 0:
                    occ[k] += (sign == 0).sum()
                else:
                    occ[k] += ((sign == np.sign(v)) & (ish == (abs(v) == 2))).sum()
            rat = hi / lo
            ratios.append(rat)
            t_u4 = t_u8 = t_two = t_eq = 0
            for a_, b_ in zip(lo, hi):
                fr = Fraction(float(b_)) / Fraction(float(a_))
                t_two += fr == 2; t_eq += fr == 1
                small = fr.limit_denominator(127)
                if small == fr:
                    if fr.numerator <= 7: t_u4 += 1
                    if fr.numerator <= 127: t_u8 += 1
            uni4 += t_u4; uni8 += t_u8; two += t_two; eq += t_eq; rows_total += o
            per.append({"name": e["n"], "shape": e["shape"], "zero_frac": float((sign == 0).mean()), "hi_frac": float((ish & (sign != 0)).mean()),
                        "ratio_min": float(rat.min()), "ratio_median": float(np.median(rat)), "ratio_max": float(rat.max()),
                        "rows_exact_4bit_grid": t_u4, "rows_exact_8bit_grid": t_u8, "rows": o})
    rat = np.concatenate(ratios)
    summ = {"container_kinds": kinds, "five_value_tensors": n_fv, "five_value_params": params, "rows": rows_total,
            "rows_hi_eq_2lo": two, "rows_hi_eq_lo": eq, "rows_exact_on_4bit_grid": uni4, "rows_exact_on_8bit_grid": uni8,
            "rows_with_no_nonzero": zero_rows,
            "level_occupancy_-hi_-lo_0_lo_hi": (occ / occ.sum()).round(4).tolist(),
            "hi_over_lo_percentiles_0_1_10_50_90_99_100": np.percentile(rat, [0, 1, 10, 50, 90, 99, 100]).round(4).tolist(),
            "per_tensor": per}
    (ROOT / "work").mkdir(exist_ok=True)
    (ROOT / "work" / "weights_census.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_tensor"}, indent=1))


if __name__ == "__main__":
    main()
