"""GENCO datakit checkpoint (base, seed0) on its own datakit test split: accuracy, timing, PYPOWER export.
Usage: python scripts/run_genco_datakit.py --case case14_ieee
"""
import argparse, json, platform, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from pfsb.models import genco_datakit as G
from pfsb.metrics import sample_errors
from pfsb.timing import time_fn
from pfsb.export import export_scenarios, max_mismatch
from torch_geometric.data import Batch

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--no-timing", action="store_true")
a = ap.parse_args()
tag = f"genco_datakit_{a.case}_base_seed0"
res = root / "results/models" / tag; res.mkdir(parents=True, exist_ok=True)
dev = torch.device(f"cuda:{a.gpu}")
raw = str(root / f"cache/hf/pf_small_{a.case}")
cfg, args, norm, task = G.load_model(str(root / f"cache/hf/genco-pf-datakit/{a.case}/base/seed0"), a.case, dev)
ids_all, ntot = G.test_scenario_ids(raw, cfg)
rng = np.random.default_rng(0)
ids = sorted(int(i) for i in (rng.choice(sorted(ids_all), a.n, replace=False) if len(ids_all) > a.n else ids_all))
print(f"{a.case}: dataset {ntot} scenarios, test {len(ids_all)}, scored {len(ids)}", flush=True)
tabs = G.read_scenarios(raw, ids)
case = export_scenarios(tabs, ids, 100.0)
items = G.build_items(args, norm, tabs, ids)
sid, VM, VA = [], [], []
for k in range(0, len(items), 64):
    s, vm, va = G.predict(task, Batch.from_data_list(items[k:k + 64]).to(dev)); sid.append(s); VM.append(vm); VA.append(va)
sid = np.concatenate(sid); VM = np.concatenate(VM); VA = np.concatenate(VA)
assert list(sid) == ids
y0 = items[0]["bus"].y
assert np.abs(y0[:, 3].numpy() - case["label_vm"][0]).max() < 1e-5 and np.abs(np.rad2deg(y0[:, 4].numpy()) - case["label_va_deg"][0]).max() < 1e-3
btype = case["bus"][:, :, 1].astype(int); slack = np.argmax(btype == 3, axis=1)
err = sample_errors(VM, VA, case["label_vm"], case["label_va_deg"], slack, btype)
pd.DataFrame({"dataset": a.case, "sample_id": ids, "condition": case["condition"], **err}).to_csv(res / "per_sample.csv", index=False)
case["pred_vm"], case["pred_va_deg"] = VM, VA
print(pd.DataFrame(err).mean().to_string(), flush=True)
chk = [max_mismatch(case, k) for k in range(0, len(ids), max(1, len(ids) // 100))]
cd = root / "cache/datasets"; cd.mkdir(parents=True, exist_ok=True)
p = cd / f"{a.case}_{tag}.npz"; np.savez_compressed(p, **case)
mp = cd / "manifest.json"; m_ = json.load(open(mp)) if mp.exists() else {}
m_[f"{a.case}:{tag}"] = dict(path=str(p), n_samples=len(ids), baseMVA=100.0, export_selfcheck_max_mismatch_mva=max(chk))
json.dump(m_, open(mp, "w"), indent=1); print("selfcheck", max(chk), flush=True)
if not a.no_timing:
    rows, gname = [], torch.cuda.get_device_name(dev)
    def bench(n, device, thr=None):
        if thr: torch.set_num_threads(thr)
        b = Batch.from_data_list([items[i % len(items)] for i in range(n)]).to(device); task.to(device)
        return time_fn(lambda: G.forward(task, b), device)
    for n in (1, 64, 1024):
        try: t = bench(n, dev)
        except torch.cuda.OutOfMemoryError: t = float("nan"); torch.cuda.empty_cache()
        rows.append(dict(device=gname, threads="", batch=n, sec_per_batch=t, ms_per_sample=1e3 * t / n)); print(rows[-1], flush=True)
    t = bench(1, torch.device("cpu"), 1)
    rows.append(dict(device="cpu", threads=1, batch=1, sec_per_batch=t, ms_per_sample=1e3 * t))
    pd.DataFrame(rows).to_csv(res / "timing.csv", index=False)
json.dump(dict(ckpt=f"gridfm/genco-pf-datakit {a.case}/base/seed0/best_model_state_dict.pt", hf_rev="f5eb20ee798b9847b723c91874af85d32a96a468",
               n_test_total=len(ids_all), n_dataset=ntot, sample_ids=ids, host=platform.node(), torch=torch.__version__),
          open(res / "run_meta.json", "w"))
