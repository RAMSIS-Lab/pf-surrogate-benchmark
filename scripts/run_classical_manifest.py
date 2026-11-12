"""Run the classical ladder on every exported-scenario entry of cache/datasets/manifest.json not yet scored.

    PFSB_... not needed.   nice -n 5 python scripts/run_classical_manifest.py [--manifest cache/datasets/manifest.json]

Entry key "<dataset>:<tag>" -> results/classical/manifest/per_sample_<dataset>_<tag>.csv (same columns as
per_sample.csv, `grid`=dataset, `condition`=the scenario's condition string, plus `tag`); an entry is skipped when that
file exists. results/classical/summary_manifest.csv is rebuilt over all entries (median and mean per dataset x tag x
condition-type x method). Use --force to rescore an entry. --max-samples N limits samples per entry (default all).
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pfsb  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pfsb.baselines import classical as C  # noqa: E402

JOBS = [("dc", None), ("dc_fd", 1), ("dc_fd", 2), ("dc_fd", 5), ("dc_fd", 10), ("dc_fd_conv", None),
        ("nr_cold", None), ("nr_dc", None)]
COLS = ["dataset", "tag", "condition", "sample_id", "method", "k", "vm_mae", "va_mae_deg", "vm_max", "va_max_deg",
        "converged", "iters", "same_root", "t_build_ms", "t_solve_ms", "load_avg", "label_mismatch", "label_flag"]


def run_entry(key, info, out, reps, max_samples):
    dataset, tag = key.split(":", 1)
    z = np.load(info["path"], allow_pickle=True)
    S = len(z["sample_ids"]) if max_samples is None else min(max_samples, len(z["sample_ids"]))
    base = float(z["baseMVA"])
    cond = z["condition"] if "condition" in z.files else np.array(["base"] * S)
    rows = []
    for i in range(S):
        ppc = {"baseMVA": base, "bus": z["bus"][i], "gen": z["gen"][i], "branch": z["branch"][i]}
        label = {"vm": z["label_vm"][i], "va_deg": z["label_va_deg"][i]}
        try:
            net = C.parse_ppc(ppc, label)
        except AssertionError as e:
            print("skip sample", z["sample_ids"][i], e, flush=True)
            continue
        lm = C.label_mismatch(net)
        for m, k in JOBS:
            r = C.run_method(net, m, k, reps)
            rows.append([dataset, tag, str(cond[i]), int(z["sample_ids"][i]), m, "" if k is None else k, r["vm_mae"],
                         r["va_mae_deg"], r["vm_max"], r["va_max_deg"], r["converged"], r["iters"], r["same_root"],
                         r["t_build_ms"], r["t_solve_ms"], os.getloadavg()[0], lm, lm > 1e-6])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(rows)


def summarize(d: Path, dest: Path):
    df = pd.concat([pd.read_csv(f) for f in sorted(d.glob("per_sample_*.csv"))], ignore_index=True)
    df["cond_type"] = df["condition"].astype(str).str.replace(r"=.*", "", regex=True)
    df["k"] = df["k"].fillna(-1)
    s = df.groupby(["dataset", "tag", "cond_type", "method", "k"]).agg(
        n=("sample_id", "count"), vm_mae_mean=("vm_mae", "mean"), vm_mae_median=("vm_mae", "median"),
        va_mae_deg_mean=("va_mae_deg", "mean"), va_mae_deg_median=("va_mae_deg", "median"),
        vm_max_max=("vm_max", "max"), va_max_deg_max=("va_max_deg", "max"), converged_frac=("converged", "mean"),
        same_root_frac=("same_root", "mean"), iters_mean=("iters", "mean"),
        t_build_ms_median=("t_build_ms", "median"), t_build_ms_mean=("t_build_ms", "mean"),
        t_solve_ms_median=("t_solve_ms", "median"), t_solve_ms_mean=("t_solve_ms", "mean"),
        load_avg_mean=("load_avg", "mean"), label_flagged=("label_flag", "sum")).reset_index()
    s["k"] = s["k"].replace(-1, np.nan)
    s.to_csv(dest, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="cache/datasets/manifest.json")
    ap.add_argument("--out", default="results/classical")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) / "manifest"
    out.mkdir(parents=True, exist_ok=True)
    for key, info in json.load(open(a.manifest)).items():
        f = out / f"per_sample_{re.sub(r'[^A-Za-z0-9_.-]', '_', key.replace(':', '_'))}.csv"
        if f.exists() and not a.force:
            print("already scored:", key)
            continue
        run_entry(key, info, f, a.reps, a.max_samples)
        print("scored", key, flush=True)
    if list(out.glob("per_sample_*.csv")):
        summarize(out, Path(a.out) / "summary_manifest.csv")


if __name__ == "__main__":
    main()
