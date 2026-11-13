"""Angle MAE against outage condition (N, N-1, N-2) on PF-Delta IEEE 118: GENCO's three released seeds (points and
their mean) against DC + k fast-decoupled steps on the same samples. Reads results/models/summary.csv and
results/classical/manifest/per_sample_pfdelta_task1.3_case118_<cond>_feasible.csv.

Usage: python scripts/fig_outage_trend.py --out results/figures/outage_trend.pdf
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "legend.fontsize": 7})
CONDS = [("n", "N"), ("n-1", "N-1"), ("n-2", "N-2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "figures" / "outage_trend.pdf"))
    a = ap.parse_args()
    m = pd.read_csv(R / "models" / "summary.csv")
    x = list(range(len(CONDS)))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for k, mk, ls in [(1, "v", ":"), (2, "o", "-"), (5, "s", "--")]:
        ys = []
        for c, _ in CONDS:
            d = pd.read_csv(R / "classical" / "manifest" / f"per_sample_pfdelta_task1.3_case118_{c}_feasible.csv")
            ys.append(d[(d.method == "dc_fd") & (d.k == k)].va_mae_deg.mean())
        ax.plot(x, ys, ls, marker=mk, color="0.35", ms=3, lw=0.9, label=f"DC + {k} FD step" + ("s" if k > 1 else ""))
    seeds = []
    for s in (1, 2, 3):
        ys = [float(m[(m.model == f"genco_pfdelta_t1.3_seed{s}") & (m.subset == f"case118_{c}_feasible")].va_mae_deg.iloc[0])
              for c, _ in CONDS]
        seeds.append(ys)
        ax.plot(x, ys, "o", color="C3", ms=2.5, alpha=0.45, label="GENCO, each released seed" if s == 1 else None)
    mean = [sum(v) / 3 for v in zip(*seeds)]
    ax.plot(x, mean, "-o", color="C3", ms=3.5, lw=1.2, label="GENCO, mean of 3 seeds")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in CONDS])
    ax.set_xlim(-0.25, len(CONDS) - 0.75)
    ax.set_xlabel("outage condition, PF-Delta IEEE 118")
    ax.set_ylabel("angle MAE (deg)")
    ax.grid(True, which="major", lw=0.3, color="0.85")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
    fig.tight_layout()
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("wrote", out, "mean", [round(v, 3) for v in mean])


if __name__ == "__main__":
    main()
