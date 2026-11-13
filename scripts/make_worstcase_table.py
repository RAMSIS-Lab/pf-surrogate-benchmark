"""Worst-case angle error (deg) over each test set: the learned model's largest per-bus angle error against DC, DC+FD2
and DC+FD5 on the same samples. Same rows as the main table, away from voltage collapse. From results/ files only:
model maxima from results/models/summary.csv (va_max_deg; for PF-Delta 118 the largest over the three seeds,
results/models/genco_pfdelta_t1.3_seed_mean.csv va_max_deg_max); classical maxima from the per-sample manifest files.

Usage: python scripts/make_worstcase_table.py -> results/table_worstcase.tex, results/table_worstcase.csv
"""
import pathlib

import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
MAN = R / "classical" / "manifest"
ROWS = [  # label, model summary key (model, subset) or seed-mean subset, classical per-sample file
    ("GENCO", "IEEE 14", ("genco_datakit_case14_ieee_base_seed0", "all"), "per_sample_case14_ieee_genco_datakit_case14_ieee_base_seed0.csv"),
    ("", "IEEE 30", ("genco_datakit_case30_ieee_base_seed0", "all"), "per_sample_case30_ieee_genco_datakit_case30_ieee_base_seed0.csv"),
    ("", "IEEE 57", ("genco_datakit_case57_ieee_base_seed0", "all"), "per_sample_case57_ieee_genco_datakit_case57_ieee_base_seed0.csv"),
    ("", "IEEE 118", ("genco_datakit_case118_ieee_base_seed0", "all"), "per_sample_case118_ieee_genco_datakit_case118_ieee_base_seed0.csv"),
    ("", "GOC 500", ("genco_datakit_case500_goc_base_seed0", "all"), "per_sample_case500_goc_genco_datakit_case500_goc_base_seed0.csv"),
    ("", "PF-Delta 118 N", "case118_n_feasible", "per_sample_pfdelta_task1.3_case118_n_feasible.csv"),
    ("", "PF-Delta 118 N-1", "case118_n-1_feasible", "per_sample_pfdelta_task1.3_case118_n-1_feasible.csv"),
    ("", "PF-Delta 118 N-2", "case118_n-2_feasible", "per_sample_pfdelta_task1.3_case118_n-2_feasible.csv"),
    ("PowerFlowNet", "IEEE 14", ("pfn_original_14", "all"), "per_sample_pfn_14_pfn_original_14.csv"),
    ("", "IEEE 118", ("pfn_original_118", "all"), "per_sample_pfn_118_pfn_original_118.csv"),
]


def fmt(x):
    return f"{x:.0f}" if x >= 10 else (f"{x:#.2g}".rstrip(".") if x >= 0.01 else f"{x:.1e}".replace("e-0", "e-"))


def main():
    m = pd.read_csv(R / "models" / "summary.csv")
    sm = pd.read_csv(R / "models" / "genco_pfdelta_t1.3_seed_mean.csv")
    out, lines = [], []
    for model, data, key, f in ROWS:
        if isinstance(key, tuple):
            g = m[(m.model == key[0]) & (m.subset == key[1])]
            mx = float(g.va_max_deg.iloc[0])
        else:
            mx = float(sm[sm.subset == key].va_max_deg_max.iloc[0])
        d = pd.read_csv(MAN / f)
        cl = {}
        for meth, k, lab in [("dc", None, "dc"), ("dc_fd", 2.0, "fd2"), ("dc_fd", 5.0, "fd5")]:
            e = d[(d.method == meth) & ((d.k == k) if k is not None else d.k.isna())]
            cl[lab] = float(e.va_max_deg.max())
        out.append(dict(model=model, data=data, model_max=mx, **cl))
        if model and lines:
            lines.append(r"\midrule")
        lines.append(" & ".join([model, data, fmt(mx), fmt(cl["dc"]), fmt(cl["fd2"]), fmt(cl["fd5"])]) + r" \\")
    pd.DataFrame(out).to_csv(R / "table_worstcase.csv", index=False)
    (R / "table_worstcase.tex").write_text("\n".join(lines) + "\n")
    print(pd.DataFrame(out).to_string())


if __name__ == "__main__":
    main()
