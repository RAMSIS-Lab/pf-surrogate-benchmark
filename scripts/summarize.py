"""Build results/models/summary.csv from every results/models/*/per_sample.csv (and timing.csv if present)."""
import sys
from pathlib import Path
import pandas as pd
root = Path(__file__).resolve().parents[1] / "results/models"
rows = []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    f = d / "per_sample.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f)
    df["kind"] = df.condition.map(lambda c: "base" if c == "base" else ("branch_out" if "branch_out" in c else "gen_out"))
    for name, g in [("all", df)] + [(k, g) for k, g in df.groupby("kind")]:
        r = dict(model=d.name, dataset=df.dataset.iloc[0], subset=name, n=len(g),
                 vm_mae=g.vm_mae.mean(), va_mae_deg=g.va_mae_deg.mean(), vm_max=g.vm_max.max(), va_max_deg=g.va_max_deg.max(),
                 vm_mae_pq=g.vm_mae_pq.mean(), va_mae_deg_pq=g.va_mae_deg_pq.mean(),
                 vm_mae_pv=g.vm_mae_pv.mean(), va_mae_deg_pv=g.va_mae_deg_pv.mean())
        t = d / "timing.csv"
        if name == "all" and t.exists():
            for _, x in pd.read_csv(t).iterrows():
                key = ("cpu1_b1" if x.device == "cpu" else f"gpu_b{int(x.batch)}") + "_ms_per_sample"
                r[key] = x.ms_per_sample
        rows.append(r)
pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
print(pd.DataFrame(rows).to_string())
