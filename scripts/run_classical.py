"""Classical baselines over PF-Delta task-1.3 test subsets. Writes results/classical/{per_sample,summary}.csv.

    PFSB_PFDELTA_ROOTS=/path/a:/path/b nice -n 5 python scripts/run_classical.py [--grids case14 ...] [--max-samples 500]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pfsb  # noqa: E402,F401  (pins threads first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pfsb.baselines import classical as C  # noqa: E402
from pfsb.data import pfdelta  # noqa: E402

COLS = ["grid", "condition", "sample_id", "method", "k", "vm_mae", "va_mae_deg", "vm_max", "va_max_deg", "converged",
        "iters", "same_root", "t_build_ms", "t_solve_ms", "load_avg", "label_mismatch", "label_flag"]
JOBS = [("dc", None), ("dc_fd", 1), ("dc_fd", 2), ("dc_fd", 5), ("dc_fd", 10), ("dc_fd_conv", None),
        ("nr_cold", None), ("nr_dc", None)]


def summarize(out: Path):
    df = pd.read_csv(out / "per_sample.csv")
    df["k"] = df["k"].fillna(-1)
    g = df.groupby(["grid", "condition", "method", "k"])
    s = g.agg(n=("sample_id", "count"), vm_mae_mean=("vm_mae", "mean"), vm_mae_median=("vm_mae", "median"),
              va_mae_deg_mean=("va_mae_deg", "mean"), va_mae_deg_median=("va_mae_deg", "median"),
              vm_max_max=("vm_max", "max"), va_max_deg_max=("va_max_deg", "max"),
              converged_frac=("converged", "mean"), same_root_frac=("same_root", "mean"),
              iters_mean=("iters", "mean"), t_build_ms_median=("t_build_ms", "median"),
              t_build_ms_mean=("t_build_ms", "mean"), t_solve_ms_median=("t_solve_ms", "median"),
              t_solve_ms_mean=("t_solve_ms", "mean"), load_avg_mean=("load_avg", "mean"),
              label_flagged=("label_flag", lambda x: int(x.sum() / len(JOBS) if False else x.sum()))).reset_index()
    s["k"] = s["k"].replace(-1, np.nan)
    s.to_csv(out / "summary.csv", index=False)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", default=["case14", "case30", "case57", "case118"])
    ap.add_argument("--conds", nargs="+", default=["n", "n-1", "n-2"])
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results/classical")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pth = out / "per_sample.csv"
    done = set()
    if pth.exists():
        d = pd.read_csv(pth, usecols=["grid", "condition"]).drop_duplicates()
        done = set(zip(d.grid, d.condition))
    else:
        csv.writer(open(pth, "w")).writerow(COLS)
    for grid in a.grids:
        for cond in a.conds:
            if (grid, cond) in done:
                continue
            ids, ppcs = pfdelta.load_test_set(grid, cond, a.cache, max_samples=a.max_samples)
            with open(out / f"test_ids_{grid}_{cond}.txt", "w") as fh:
                fh.write("\n".join(map(str, ids)) + "\n")
            rows = []
            for sid, ppc in zip(ids, ppcs):
                net = C.parse_ppc(ppc)
                lm = C.label_mismatch(net)
                for m, k in JOBS:
                    r = C.run_method(net, m, k, a.reps)
                    rows.append([grid, cond, sid, m, "" if k is None else k, r["vm_mae"], r["va_mae_deg"], r["vm_max"],
                                 r["va_max_deg"], r["converged"], r["iters"], r["same_root"], r["t_build_ms"],
                                 r["t_solve_ms"], os.getloadavg()[0], lm, lm > 1e-6])
            with open(pth, "a", newline="") as fh:
                csv.writer(fh).writerows(rows)
            print(f"{grid} {cond}: {len(ids)} samples done, load {os.getloadavg()[0]:.1f}", flush=True)
            summarize(out)


if __name__ == "__main__":
    main()
