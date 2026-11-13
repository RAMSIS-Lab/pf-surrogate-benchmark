"""Wall-clock time per sample (ms) by deployment, per grid: learned models on one GPU at batch 1, 64 and 1024, and the
classical methods on one CPU thread and spread over 40 CPU cores (idle Intel Xeon Gold 6148 node). Also Newton and
fast-decoupled iteration counts. Every number comes from results/ files:
  models:    results/models/<name>/timing.csv           (forward pass, batch on device, median of 3)
  classical: results/classical_pitzer/summary.csv       (1 thread, build + solve median, PF-Delta N test samples)
             results/classical_pitzer/throughput.csv    (process pool, 500 PF-Delta N samples)

Usage: python scripts/make_timing_table.py   -> results/table_timing.tex, results/table_timing.csv
"""
import pathlib

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
GRIDS = [("case14", "14"), ("case30", "30"), ("case57", "57"), ("case118", "118")]
MODELS = {"GENCO": "genco_datakit_{c}_ieee_base_seed0", "PowerFlowNet": "pfn_original_{n}"}


def gpu(name, batch):
    f = R / "models" / name / "timing.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    t = t[(t.device != "cpu") & (t.batch == batch)]
    return float(t.ms_per_sample.iloc[0]) if not t.empty else None


def fmt(x):
    if x is None or pd.isna(x):
        return "--"
    return f"{x:.0f}" if x >= 10 else (f"{x:#.2g}".rstrip(".") if x >= 0.01 else f"{x:.3f}")


def main():
    s = pd.read_csv(R / "classical_pitzer" / "summary.csv")
    thr = pd.read_csv(R / "classical_pitzer" / "throughput.csv")
    rows = []
    for c, n in GRIDS:
        for lab, pat in MODELS.items():
            name = pat.format(c=c, n=n)
            if (R / "models" / name / "timing.csv").exists():
                rows.append(dict(grid=n, method=lab + " (GPU)", one=gpu(name, 1), b64=gpu(name, 64), b1024=gpu(name, 1024),
                                 cores40=None, iters=None))
        for meth, k, lab in [("dc", None, "DC"), ("dc_fd", 2.0, "DC+FD$_2$"), ("dc_fd_conv", None, "FD to $10^{-8}$"),
                             ("nr_cold", None, "NR, flat start"), ("nr_dc", None, "NR, DC start")]:
            d = s[(s.grid == c) & (s.condition == "n") & (s.method == meth) & ((s.k == k) if k is not None else s.k.isna())]
            e = thr[(thr.grid == c) & (thr.condition == "n") & (thr.procs == 40) & (thr.method == meth)
                    & ((thr.k == k) if k is not None else thr.k.isna())]
            one = float(d.t_build_ms_median.iloc[0] + d.t_solve_ms_median.iloc[0]) if not d.empty else None
            c40 = 1000.0 / float(e.samples_per_s.iloc[0]) if not e.empty else None
            it = float(d.iters_mean.iloc[0]) if (not d.empty and meth in ("dc_fd_conv", "nr_cold", "nr_dc")) else None
            rows.append(dict(grid=n, method=lab + " (CPU)", one=one, b64=None, b1024=None, cores40=c40, iters=it))
    df = pd.DataFrame(rows)
    df.to_csv(R / "table_timing.csv", index=False)
    # compact layout for the paper: rows = methods, columns = IEEE 14 and 118 (single sample | bulk), iterations at 118
    order = ["GENCO (GPU)", "PowerFlowNet (GPU)", "DC (CPU)", "DC+FD$_2$ (CPU)", "FD to $10^{-8}$ (CPU)",
             "NR, flat start (CPU)", "NR, DC start (CPU)"]
    comp = []
    for mth in order:
        cells = [mth]
        for g in ("14", "118"):
            r = df[(df.grid == g) & (df.method == mth)]
            if r.empty:
                cells += ["--", "--"]; continue
            r = r.iloc[0]
            bulk = r.b1024 if "GPU" in mth else r.cores40
            cells += [fmt(r.one), fmt(bulk)]
        r = df[(df.grid == "118") & (df.method == mth)]
        it = r.iters.iloc[0] if not r.empty else None
        cells.append("--" if it is None or pd.isna(it) else f"{it:.1f}")
        comp.append(" & ".join(cells) + r" \\")
    (R / "table_timing_compact.tex").write_text("\n".join(comp) + "\n")
    lines, prev = [], None
    for r in df.itertuples():
        if prev is not None and r.grid != prev:
            lines.append(r"\midrule")
        lines.append(" & ".join([r.grid if r.grid != prev else "", r.method, fmt(r.one), fmt(r.b64), fmt(r.b1024),
                                 fmt(r.cores40), "--" if r.iters is None or pd.isna(r.iters) else f"{r.iters:.1f}"]) + r" \\")
        prev = r.grid
    (R / "table_timing.tex").write_text("\n".join(lines) + "\n")
    print(df.to_string())


if __name__ == "__main__":
    main()
