"""Speed vs accuracy figure: angle MAE (deg) against time per sample (ms), log-log, one panel per dataset.

Classical ladder (single-thread CPU, matrix build + solve, median over samples) as a connected line
DC -> DC+FD1 -> FD2 -> FD5 -> FD10, with NR (flat) and NR (DC) as separate markers. Each learned model is a point
per deployment (GPU batch 1, GPU batch 1024). Reads results/classical/summary.csv and results/models/summary.csv.

Usage: python scripts/fig_speed_accuracy.py --panels "case118:n:PF-Delta IEEE 118 (N)" ... --out figures/speed_accuracy.pdf
Each panel spec is <classical grid key>:<condition>:<title>[:<model dataset key>].
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
FLOOR = 1e-6  # angle errors below this are plotted at the floor (Newton reaches the label to ~1e-6 deg)

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.fontsize": 7})


def classical_rows(c, grid, cond):
    d = c[(c.grid == grid) & (c.condition == cond)].copy()
    d["t"] = d.t_build_ms_median + d.t_solve_ms_median
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--out", default="figures/speed_accuracy.pdf")
    ap.add_argument("--width", type=float, default=3.5)
    a = ap.parse_args()
    c = pd.read_csv(R / "classical" / "summary.csv")
    m = pd.read_csv(R / "models" / "summary.csv")
    n = len(a.panels)
    fig, axes = plt.subplots(1, n, figsize=(a.width, 2.1), squeeze=False, sharey=True)
    for ax, spec in zip(axes[0], a.panels):
        parts = spec.split(":")
        grid, cond, title = parts[0], parts[1], parts[2]
        mkey = parts[3] if len(parts) > 3 else grid
        d = classical_rows(c, grid, cond)
        lad = [("dc", None, "DC")] + [("dc_fd", k, f"FD{k}") for k in (1, 2, 5, 10)]
        xs, ys = [], []
        for meth, k, lab in lad:
            r = d[(d.method == meth) & ((d.k == k) if k is not None else d.k.isna())]
            if r.empty:
                continue
            x, y = float(r.t.iloc[0]), max(float(r.va_mae_deg_mean.iloc[0]), FLOOR)
            xs.append(x); ys.append(y)
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(3, 2), fontsize=7)
        ax.plot(xs, ys, "-o", color="0.25", ms=3, lw=0.8, label="DC + $k$ FD (CPU)")
        for meth, mk, lab in [("nr_cold", "s", "NR flat (CPU)"), ("nr_dc", "D", "NR from DC (CPU)")]:
            r = d[d.method == meth]
            if not r.empty:
                ax.plot(float(r.t.iloc[0]), max(float(r.va_mae_deg_mean.iloc[0]), FLOOR), mk,
                        color="0.55", ms=3.5, mfc="none", label=lab)
        mm = m[(m.dataset == mkey) & (m.subset.isin(["all", cond]))]
        for i, (name, g) in enumerate(mm.groupby("model")):
            g = g.iloc[0]
            col = f"C{i}"
            for col_t, mk, suf in [("gpu_b1_ms_per_sample", "o", "GPU b1"), ("gpu_b1024_ms_per_sample", "^", "GPU b1024")]:
                if pd.notna(g.get(col_t)):
                    ax.plot(g[col_t], g.va_mae_deg, mk, color=col, ms=4, label=f"{name} ({suf})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("time per sample (ms)")
        ax.grid(True, which="major", lw=0.3, color="0.85")
    axes[0][0].set_ylabel("angle MAE (deg)")
    h, l = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        hh, ll = ax.get_legend_handles_labels()
        for x, y in zip(hh, ll):
            if y not in l:
                h.append(x); l.append(y)
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
