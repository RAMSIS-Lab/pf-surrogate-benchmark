"""Multi-process throughput (samples/s, build + solve per sample) of the classical methods for P = 1 and 16 workers.

    PFSB_PFDELTA_ROOTS=... nice -n 5 python scripts/run_throughput.py
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pfsb  # noqa: E402,F401
from pfsb.baselines import classical as C  # noqa: E402
from pfsb.data import pfdelta  # noqa: E402

JOBS = [("dc", None), ("dc_fd", 2), ("dc_fd_conv", None), ("nr_cold", None), ("nr_dc", None)]

ap = argparse.ArgumentParser()
ap.add_argument("--grids", nargs="+", default=["case14", "case30", "case57", "case118"])
ap.add_argument("--conds", nargs="+", default=["n", "n-1", "n-2"])
ap.add_argument("--procs", nargs="+", type=int, default=[1, 16])
ap.add_argument("--max-samples", type=int, default=500)
ap.add_argument("--out", default="results/classical/throughput.csv")
a = ap.parse_args()
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["grid", "condition", "method", "k", "procs", "n_samples", "samples_per_s", "load_avg"])
    for g in a.grids:
        for c in a.conds:
            _, ppcs = pfdelta.load_test_set(g, c, "cache", max_samples=a.max_samples)
            nets = [C.parse_ppc(p) for p in ppcs]
            for m, k in JOBS:
                for P in a.procs:
                    sps = C.throughput(nets, m, k, P)
                    w.writerow([g, c, m, "" if k is None else k, P, len(nets), f"{sps:.1f}", os.getloadavg()[0]])
                    fh.flush()
            print(g, c, "done", flush=True)
