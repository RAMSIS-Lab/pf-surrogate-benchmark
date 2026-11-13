"""Build the main results table (LaTeX rows + a CSV) from results/ so no number is copied by hand.

Rows: GENCO datakit grids (seed0), GENCO PF-Delta IEEE 118 (mean of seeds 1-3, per subset), and the PF-Delta
retrained-model rows (classical columns filled from results/classical, model columns from results/models/<name>_<grid>
when present, else a placeholder). Classical errors are means over the same samples as the model; classical times are
medians of (matrix build + solve) per sample from --timing-root (default results/classical).

Usage: python scripts/make_table.py [--timing-root results/classical_pitzer] --out-tex ../paper/table_main.tex
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
METH = [("dc", None), ("dc_fd", 1), ("dc_fd", 2), ("dc_fd", 5), ("dc_fd", 10), ("nr_cold", None)]


def cl_stats(ps):
    """ps: classical per-sample frame for one sample set -> {(method,k): (va_mean, vm_mean, t_med_ms)}."""
    out = {}
    for m, k in METH:
        d = ps[(ps.method == m) & ((ps.k == k) if k is not None else ps.k.isna())]
        if d.empty:
            continue
        out[(m, k)] = (d.va_mae_deg.mean(), d.vm_mae.mean(), float(np.median(d.t_build_ms + d.t_solve_ms)),
                       d.converged.mean() if "converged" in d else np.nan)
    return out


def timing_override(root, grid_key, fname=None):
    """Median per-sample time per method from a quiet-node run: the same sample file if given, else by grid key."""
    out = {}
    if fname is not None and (root / "manifest" / fname).exists():
        d = pd.read_csv(root / "manifest" / fname)
        for m, k in METH:
            e = d[(d.method == m) & ((d.k == k) if k is not None else d.k.isna())]
            if not e.empty:
                out[(m, k)] = float(np.median(e.t_build_ms + e.t_solve_ms))
        return out
    for f in [root / "per_sample.csv"] + sorted((root / "manifest").glob("per_sample_*.csv")) if root.exists() else []:
        if not f.exists():
            continue
        d = pd.read_csv(f)
        key = "grid" if "grid" in d else "dataset"
        d = d[d[key] == grid_key]
        for m, k in METH:
            e = d[(d.method == m) & ((d.k == k) if k is not None else d.k.isna())]
            if not e.empty:
                out[(m, k)] = float(np.median(e.t_build_ms + e.t_solve_ms))
        if out:
            break
    return out


def fmt(x, kind):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return r"\PH{--}"
    if kind == "va":
        if x >= 10:
            return f"{x:.0f}"
        return f"{x:#.2g}".rstrip(".") if x >= 0.01 else f"{x:.1e}".replace("e-0", "e-")
    if kind == "vm":
        return f"{x:.1e}".replace("e-0", "e-")
    if kind == "t":
        return f"{x:#.2g}".rstrip(".") if x < 10 else f"{x:.0f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing-root", default=str(R / "classical"))
    ap.add_argument("--out-tex", default=str(R / "table_main.tex"))
    ap.add_argument("--out-csv", default=str(R / "table_main.csv"))
    a = ap.parse_args()
    troot = pathlib.Path(a.timing_root)
    msum = pd.read_csv(R / "models" / "summary.csv")
    rows = []

    def add(data, model, vm, va, tb1, tbig, ps, tkey, fname=None):
        st = cl_stats(ps)
        tov = timing_override(troot, tkey, fname) if troot != R / "classical" else {}
        t = {mk: tov.get(mk, st[mk][2]) for mk in st}
        rows.append(dict(data=data, model=model, vm=vm, va=va, gpu_b1=tb1, gpu_bmax=tbig,
                         **{f"va_{m}{k or ''}": st[(m, k)][0] for (m, k) in st},
                         **{f"vm_{m}{k or ''}": st[(m, k)][1] for (m, k) in st},
                         **{f"t_{m}{k or ''}": t[(m, k)] for (m, k) in st},
                         nr_cold_conv=st.get(("nr_cold", None), (0, 0, 0, np.nan))[3]))

    # GENCO datakit grids
    for case, lab in [("case14_ieee", "IEEE 14"), ("case30_ieee", "IEEE 30"), ("case57_ieee", "IEEE 57"),
                      ("case118_ieee", "IEEE 118"), ("case500_goc", "GOC 500")]:
        name = f"genco_datakit_{case}_base_seed0"
        f = R / "classical" / "manifest" / f"per_sample_{case}_{name}.csv"
        g = msum[(msum.model == name) & (msum.subset == "all")]
        if g.empty or not f.exists():
            continue
        g = g.iloc[0]
        big = g.gpu_b1024_ms_per_sample if pd.notna(g.gpu_b1024_ms_per_sample) else np.nan
        tm = pd.read_csv(R / "models" / name / "timing.csv")
        gpu = tm[tm.device != "cpu"]
        big = float(gpu.ms_per_sample[gpu.batch == gpu.batch.max()].iloc[0])
        add(f"datakit {lab}", "GENCO", g.vm_mae, g.va_mae_deg, g.gpu_b1_ms_per_sample, big, pd.read_csv(f), case.split("_")[0])

    # GENCO PF-Delta 118, seed mean per subset
    sm = pd.read_csv(R / "models" / "genco_pfdelta_t1.3_seed_mean.csv")
    tm = pd.read_csv(R / "models" / "genco_pfdelta_t1.3_seed1" / "timing.csv")
    gpu = tm[tm.device != "cpu"]
    tb1 = float(gpu.ms_per_sample[gpu.batch == 1].iloc[0]); tbig = float(gpu.ms_per_sample[gpu.batch == gpu.batch.max()].iloc[0])
    for sub, lab in [("case118_n_feasible", "N"), ("case118_n-1_feasible", "N-1"), ("case118_n-2_feasible", "N-2"),
                     ("case118_n_nose", "N near collapse"), ("case118_n-1_nose", "N-1 near collapse"), ("case118_n-2_nose", "N-2 near collapse")]:
        f = R / "classical" / "manifest" / f"per_sample_pfdelta_task1.3_{sub}.csv"
        r = sm[sm.subset == sub]
        if r.empty or not f.exists():
            continue
        r = r.iloc[0]
        add(f"PF-Delta 118 {lab}", "GENCO (3 seeds)", r.vm_mae_mean, r.va_mae_deg_mean, tb1, tbig, pd.read_csv(f), "case118", f.name)

    # Other released-weight models (PowerFlowNet authors' weights, SafePowerGraph): any summary row whose model starts
    # with a known prefix, paired with the classical manifest file that names the same model.
    for prefix, label in [("pfn_original_", "PowerFlowNet"), ("spg_", "SafePowerGraph")]:
        for name in sorted(set(msum.model[msum.model.str.startswith(prefix)]), key=lambda x: int("".join(c for c in x[len(prefix):].split("_")[0] if c.isdigit()) or 0)):
            if any(t in name for t in ("nodrop", "large")):
                continue
            g = msum[(msum.model == name) & (msum.subset == "all")]
            if g.empty:
                g = msum[msum.model == name].head(1)
            files = sorted((R / "classical" / "manifest").glob(f"per_sample_*{name}*.csv"))
            if g.empty or not files:
                continue
            g = g.iloc[0]
            tmf = R / "models" / name / "timing.csv"
            tb1, tbig = g.get("gpu_b1_ms_per_sample", np.nan), np.nan
            if tmf.exists():
                tm = pd.read_csv(tmf); gpu = tm[tm.device != "cpu"]
                if not gpu.empty:
                    tb1 = float(gpu.ms_per_sample[gpu.batch == gpu.batch.min()].iloc[0])
                    tbig = float(gpu.ms_per_sample[gpu.batch == gpu.batch.max()].iloc[0])
            grid = "case" + "".join(ch for ch in name[len(prefix):].split("_")[0] if ch.isdigit())
            sys_name = "RTE" if "rte" in name else "IEEE"
            add(f"{label} authors' {sys_name} {grid[4:]}", label, g.vm_mae, g.va_mae_deg, tb1, tbig, pd.read_csv(files[0]), grid)

    df = pd.DataFrame(rows)
    df.to_csv(a.out_csv, index=False)
    lines = []
    prev = None
    for r in df.itertuples():
        mname = r.model.replace(" (3 seeds)", "")
        if prev is not None and mname != prev:
            lines.append(r"\midrule")
        shown = mname if mname != prev else ""
        prev = mname
        data = r.data.replace("PowerFlowNet ", "").replace("SafePowerGraph ", "")
        lines.append(" & ".join([shown, data, fmt(r.vm, "vm"), fmt(r.va, "va"),
                                 f"{fmt(r.gpu_b1, 't')} / {fmt(r.gpu_bmax, 't')}",
                                 fmt(r.va_dc, "va"), fmt(r.va_dc_fd1, "va"), fmt(r.va_dc_fd2, "va"), fmt(r.va_dc_fd5, "va"),
                                 fmt(r.vm_dc, "vm"), fmt(r.vm_dc_fd1, "vm"), fmt(r.vm_dc_fd2, "vm"), fmt(r.vm_dc_fd5, "vm"),
                                 fmt(r.t_dc_fd2, "t"),
                                 fmt(r.t_nr_cold, "t") if not (r.nr_cold_conv < 0.5) else "div."]) + r" \\")
    pathlib.Path(a.out_tex).write_text("\n".join(lines) + "\n")
    print(df[["data", "model", "va", "va_dc", "va_dc_fd1", "va_dc_fd2", "va_dc_fd5", "t_dc_fd2", "t_nr_cold"]].to_string())


if __name__ == "__main__" and "--retrained" not in __import__("sys").argv:
    main()


def retrained_table(out_tex, out_csv):
    """Table of angle MAE on PF-Delta (500 test samples per grid and condition): classical ladder and the three
    recipe-trained models. A model cell is a placeholder until results/models/<model>_pfdelta_<grid> exists.
    Rows tagged with an 'aug' run (case118 checkpoints trained 2026-08 with the same recipe) are printed in red."""
    c = pd.read_csv(R / "classical" / "summary.csv")
    m = pd.read_csv(R / "models" / "summary.csv")
    rows, lines = [], []
    for grid in ["case14", "case30", "case57", "case118"]:
        for cond, lab in [("n", "N"), ("n-1", "N-1"), ("n-2", "N-2")]:
            d = c[(c.grid == grid) & (c.condition == cond)]
            def cv(meth, k=None):
                e = d[(d.method == meth) & ((d.k == k) if k is not None else d.k.isna())]
                return float(e.va_mae_deg_mean.iloc[0]) if not e.empty else np.nan
            r = dict(grid=grid, cond=lab, dc=cv("dc"), fd1=cv("dc_fd", 1), fd2=cv("dc_fd", 2), fd5=cv("dc_fd", 5))
            cells = []
            for key in ["pfnet", "gns", "canos"]:
                g = m[(m.model == f"{key}_pfdelta_{grid}") & (m.subset == cond)]
                prov = False
                if g.empty:
                    g = m[(m.model == f"{key}_pfdelta_{grid}_aug") & (m.subset == cond)]
                    prov = not g.empty
                v = float(g.va_mae_deg.iloc[0]) if not g.empty else np.nan
                r[key] = v
                s = fmt(v, "va")
                cells.append(r"\PH{" + s + "}" if prov and "PH" not in s else s)
            rows.append(r)
            lines.append(" & ".join([grid[4:] if cond == "n" else "", lab, fmt(r["dc"], "va"), fmt(r["fd1"], "va"),
                                     fmt(r["fd2"], "va"), fmt(r["fd5"], "va")] + cells) + r" \\")
        lines.append(r"\midrule" if grid != "case118" else "")
    pathlib.Path(out_tex).write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__" and "--retrained" in __import__("sys").argv:
    retrained_table(str(R / "table_retrained.tex"), str(R / "table_retrained.csv"))
