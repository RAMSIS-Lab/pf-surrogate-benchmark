"""Evaluate a released GENCO case118 checkpoint on its own test split: accuracy, timing, PYPOWER export.
Usage: python -I scripts/run_genco.py --ckpt-dir <.../finetune/case118_250000> --data-path <.../data> --tag genco_case118_250000
"""
import argparse, json, os, platform, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfsb.models import genco
from pfsb.metrics import sample_errors
from pfsb.timing import time_fn
from pfsb.export import export_scenarios, max_mismatch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--data-path", required=True)
ap.add_argument("--network", default="case118_ieee"); ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--no-export", action="store_true"); ap.add_argument("--no-timing", action="store_true")
a = ap.parse_args()
root = Path(__file__).resolve().parents[1]
res = root / "results/models" / a.tag; res.mkdir(parents=True, exist_ok=True)
dev = torch.device(f"cuda:{a.gpu}")
dm, task = genco.load(a.ckpt_dir, a.data_path, a.network, dev)
ids_all = sorted(dm.test_scenario_ids[0])
rng = np.random.default_rng(0)
ids = sorted(int(i) for i in (rng.choice(ids_all, a.n, replace=False) if len(ids_all) > a.n else ids_all))
print("test scenarios", len(ids_all), "evaluated", len(ids), flush=True)
ds = dm.datasets[0]
from torch_geometric.data import Batch
items = [ds[i] for i in ids]
raw = Path(a.data_path) / a.network / "raw"
bus_df = pd.read_parquet(raw / "bus_data.parquet"); gen_df = pd.read_parquet(raw / "gen_data.parquet"); br_df = pd.read_parquet(raw / "branch_data.parquet")
tabs = (bus_df, gen_df, br_df)
case = export_scenarios(tabs, ids, 100.0)
# predictions in chunks of 64
sid, VM, VA = [], [], []
for k in range(0, len(items), 64):
    b = Batch.from_data_list(items[k:k + 64]).to(dev)
    s, vm, va = genco.predict(task, b); sid.append(s); VM.append(vm); VA.append(va)
sid = np.concatenate(sid); VM = np.concatenate(VM); VA = np.concatenate(VA)
assert list(sid) == ids, "scenario order mismatch"
btype = case["bus"][:, :, 1].astype(int); slack = np.argmax(btype == 3, axis=1)
err = sample_errors(VM, VA, case["label_vm"], case["label_va_deg"], slack, btype)
df = pd.DataFrame({"dataset": a.network, "sample_id": ids, "condition": case["condition"], **err})
df.to_csv(res / "per_sample.csv", index=False)
case["pred_vm"], case["pred_va_deg"] = VM, VA
print(df.describe().T[["mean", "max"]], flush=True)
if not a.no_export:
    chk = [max_mismatch(case, k) for k in range(0, len(ids), max(1, len(ids) // 100))]
    print("export self-check: max |mismatch| MVA over", len(chk), "samples:", max(chk), flush=True)
    cd = root / "cache/datasets"; cd.mkdir(parents=True, exist_ok=True)
    p = cd / f"{a.network}_{a.tag}.npz"
    np.savez_compressed(p, **case)
    mp = cd / "manifest.json"
    man = json.load(open(mp)) if mp.exists() else {}
    man[f"{a.network}:{a.tag}"] = dict(path=str(p), n_samples=len(ids), baseMVA=100.0, export_selfcheck_max_mismatch_mva=max(chk))
    json.dump(man, open(mp, "w"), indent=1)
# timing (forward pass of the network only; batches pre-built and on device)
if not a.no_timing:
    rows = []
    gname = torch.cuda.get_device_name(dev)
    def bench(n, device, thr=None):
        if thr: torch.set_num_threads(thr)
        sub = [items[i % len(items)] for i in range(n)]
        b = Batch.from_data_list(sub).to(device)
        task.to(device)
        t = time_fn(lambda: genco.forward(task, b), device)
        return t
    for n in (1, 64, 1024):
        try:
            t = bench(n, dev)
            rows.append(dict(device=gname, threads="", batch=n, sec_per_batch=t, ms_per_sample=1e3 * t / n))
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(device=gname, threads="", batch=n, sec_per_batch=float("nan"), ms_per_sample=float("nan"))); torch.cuda.empty_cache()
        print(rows[-1], flush=True)
    t = bench(1, torch.device("cpu"), 1)
    rows.append(dict(device="cpu", threads=1, batch=1, sec_per_batch=t, ms_per_sample=1e3 * t)); print(rows[-1])
    pd.DataFrame(rows).to_csv(res / "timing.csv", index=False)
json.dump(dict(ckpt=os.path.abspath(a.ckpt_dir), n_test_total=len(ids_all), sample_ids=ids, host=platform.node(),
               torch=torch.__version__), open(res / "run_meta.json", "w"))
