"""Speed vs accuracy: angle MAE (deg) against time per sample (ms), log-log, one panel per dataset row of
results/table_main.csv (written by scripts/make_table.py, so figure and table share every number).

Classical ladder (one CPU thread, matrix build + solve, median per sample): DC -> DC+FD1 -> FD2 -> FD5 -> FD10 as a
line, cold-start Newton as an open square. Learned model: filled circle at GPU batch 1, filled triangle at the largest
GPU batch (1024, or 512 where 1024 did not fit).

Usage: python scripts/fig_speed_accuracy.py --rows "datakit IEEE 14" "datakit IEEE 118" ... --out figures/speed_accuracy.pdf
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
FLOOR = 1e-5  # Newton reaches the label to below this; plotted at the floor
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})


def plain(v, _):
    return f"{v:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--titles", nargs="+")
    ap.add_argument("--out", default="figures/speed_accuracy.pdf")
    ap.add_argument("--width", type=float, default=7.16)
    ap.add_argument("--table", default=str(R / "table_main.csv"))
    ap.add_argument("--throughput", default=str(R / "classical_pitzer" / "throughput.csv"))
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--model", default="GENCO")
    a = ap.parse_args()
    t = pd.read_csv(a.table)
    tp = pathlib.Path(a.throughput)
    thr = pd.read_csv(tp) if tp.exists() else None
    n = len(a.rows)
    fig, axes = plt.subplots(1, n, figsize=(a.width, 2.15), squeeze=False, sharey=True)
    for i, (ax, row) in enumerate(zip(axes[0], a.rows)):
        r = t[t.data == row].iloc[0]
        lad = [("dc", "DC"), ("dc_fd1", "1"), ("dc_fd2", "2"), ("dc_fd5", "5"), ("dc_fd10", "10")]
        xs = [r[f"t_{m}"] for m, _ in lad]
        ys = [max(r[f"va_{m}"], FLOOR) for m, _ in lad]
        ax.plot(xs, ys, "-o", color="0.3", ms=2.8, lw=0.8, label="DC + $k$ FD steps, 1 CPU thread")
        for (x, y, (_, lab)) in zip(xs, ys, lad):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(3, 1), fontsize=7, color="0.3")
        ax.plot(r.t_nr_cold, FLOOR, "s", mfc="none", color="0.45", ms=3.5, label="Newton (flat start), 1 CPU thread")
        if thr is not None:
            grid = "case" + row.split("IEEE ")[-1].split()[0] if "IEEE" in row else "case" + row.split()[1]
            cond = {"N-1": "n-1", "N-2": "n-2"}.get(row.split()[-1], "n")
            e = thr[(thr.grid == grid) & (thr.condition == cond) & (thr.procs == a.procs)]
            for meth, k, mk, y, lab in [("dc_fd", 2.0, "^", r.va_dc_fd2, f"DC + 2 FD steps, {a.procs} CPU cores"),
                                        ("nr_cold", None, "s", FLOOR, f"Newton (flat start), {a.procs} CPU cores")]:
                q = e[(e.method == meth) & ((e.k == k) if k is not None else e.k.isna())]
                if not q.empty:
                    ax.plot(1000.0 / float(q.samples_per_s.iloc[0]), max(y, FLOOR), mk, color="0.45", ms=3.8, label=lab)
        if pd.notna(r.va):
            ax.plot(r.gpu_b1, r.va, "o", color="C3", ms=4, label=f"{a.model}, GPU batch 1")
            ax.plot(r.gpu_bmax, r.va, "^", color="C3", ms=4, label=f"{a.model}, GPU large batch")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10)); ax.xaxis.set_major_formatter(FuncFormatter(plain))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=100, numticks=8))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(a.titles[i] if a.titles else row, fontsize=8)
        ax.set_xlabel("time per sample (ms)")
        ax.grid(True, which="major", lw=0.3, color="0.85")
    lo, hi = axes[0][0].get_ylim()
    axes[0][0].set_ylim(lo, hi * 4)  # headroom so the ladder labels (e.g. "DC") stay inside every panel
    axes[0][0].set_ylabel("angle MAE (deg)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=0.6)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
