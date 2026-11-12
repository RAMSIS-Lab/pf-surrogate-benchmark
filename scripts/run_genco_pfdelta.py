"""GENCO on PF-Delta case118 task 1.3: per-subset accuracy (n, n-1, n-2 x feasible, nose), RMSE reproduction check
against the authors' mlflow CSVs, GPU/CPU timing, PYPOWER export.
Needs PYTHONPATH to contain the graphkit branch clone (genco-paper-repro-pfdelta).
"""
import argparse, glob, json, os, platform, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from pfsb.models import genco_pfdelta as G
from pfsb.metrics import sample_errors
from pfsb.timing import time_fn
from pfsb.export import export_scenarios, max_mismatch
from torch_geometric.data import Batch

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--hf", default=str(root / "cache/hf/genco-pfdelta")); ap.add_argument("--data", default=str(root / "cache/hf/pfdelta_task1.3/case118"))
ap.add_argument("--cfg", default=str(root / "cache/graphkit-pfdelta/scripts/pfdelta/config/HGNS_PF_pfdelta_bs64_seed{seed}.yaml"))
ap.add_argument("--no-timing", action="store_true"); ap.add_argument("--no-export", action="store_true")
a = ap.parse_args()
tag = f"genco_pfdelta_t1.3_seed{a.seed}"
res = root / "results/models" / tag; res.mkdir(parents=True, exist_ok=True)
dev = torch.device(f"cuda:{a.gpu}")
task, stats, dsets, gk = G.load(f"{a.hf}/task_1.3/seed{a.seed}/last.pt", a.cfg.format(seed=a.seed), a.data, dev)
print("graphkit:", gk, "stats:", {k: float(v) for k, v in stats.items()}, flush=True)
rows, repro, timing_items, man = [], [], None, {}
# authors' mlflow run for this seed
auth = {}
for rd in glob.glob(f"{a.hf}/mlflow/1.3/*/"):
    nm = open(rd + "tags/mlflow.runName").read().strip()
    if nm.endswith(f"seed{a.seed}"):
        auth = {Path(f).name[:-9]: pd.read_csv(f).set_index("Metric") for f in glob.glob(rd + "artifacts/test/*_RMSE.csv")}
for name, ds in dsets.items():
    grid, sub = name.split("_")[1:]
    tdir = f"{a.data}/test/case118/{grid}/{sub}/raw"
    tabs = tuple(pd.read_parquet(f"{tdir}/{t}_data.parquet") for t in ("bus", "gen", "branch"))
    orig = list(tabs[0].scenario.unique())      # dataset index i  <->  i-th unique scenario of the parquet
    assert len(orig) == len(ds)
    case = export_scenarios(tabs, orig, 100.0)
    items = [ds[i] for i in range(len(ds))]
    sid, VM, VA, mse = [], [], [], []
    for k in range(0, len(items), 64):
        b = Batch.from_data_list(items[k:k + 64]).to(dev)
        s, vm, va = G.predict(task, b); sid.append(s); VM.append(vm); VA.append(va)
        out = G.forward(task, b)["bus"]; y = b.y_dict["bus"]; m = b.mask_dict["PQ"]
        mse.append((len(s), ((out[m][:, :2] - y[m][:, [3, 4]]) ** 2).mean(0).cpu().numpy()))
    sid = np.concatenate(sid); VM = np.concatenate(VM); VA = np.concatenate(VA)
    assert list(sid) == list(range(len(ds)))
    assert np.abs(VM[:, :1].shape[0] - len(orig)) == 0
    # check bus order / label agreement with the dataset's own targets
    y0 = items[0]["bus"].y
    assert np.abs(y0[:, 3].numpy() - case["label_vm"][0]).max() < 1e-5 and np.abs(np.rad2deg(y0[:, 4].numpy()) - case["label_va_deg"][0]).max() < 1e-3
    btype = case["bus"][:, :, 1].astype(int); slack = np.argmax(btype == 3, axis=1)
    err = sample_errors(VM, VA, case["label_vm"], case["label_va_deg"], slack, btype)
    df = pd.DataFrame({"dataset": name, "sample_id": orig, "condition": case["condition"], **err}); rows.append(df)
    w = np.array([n for n, _ in mse]); M = np.stack([m for _, m in mse])
    rm = np.sqrt((M * w[:, None]).sum(0) / w.sum())   # per-batch MSE averaged weighted by batch size, as graphkit logs it
    ak = auth.get(f"test_{name}"); 
    repro.append(dict(subset=name, n=len(ds), ours_rmse_pq_vm=rm[0], ours_rmse_pq_va_rad=rm[1],
                      authors_rmse_pq_vm=ak.loc["RMSE-PQ", "Vm (p.u.)"] if ak is not None else np.nan,
                      authors_rmse_pq_va_rad=ak.loc["RMSE-PQ", "Va (radians)"] if ak is not None else np.nan))
    print(repro[-1], flush=True)
    if not a.no_export and a.seed == 1:
        chk = [max_mismatch(case, k) for k in range(0, len(orig), max(1, len(orig) // 50))]
        cd = root / "cache/datasets"; cd.mkdir(parents=True, exist_ok=True)
        case["pred_vm_seed1"], case["pred_va_deg_seed1"] = VM, VA
        p = cd / f"pfdelta1.3_{name}.npz"; np.savez_compressed(p, **case)
        mp = cd / "manifest.json"; m_ = json.load(open(mp)) if mp.exists() else {}
        m_[f"pfdelta_task1.3:{name}"] = dict(path=str(p), n_samples=len(orig), baseMVA=100.0, export_selfcheck_max_mismatch_mva=max(chk))
        json.dump(m_, open(mp, "w"), indent=1)
        print("selfcheck", name, max(chk), flush=True)
    if name == "case118_n-1_feasible":
        timing_items = items
pd.concat(rows).to_csv(res / "per_sample.csv", index=False)
pd.DataFrame(repro).to_csv(res / "repro_check.csv", index=False)
if not a.no_timing:
    trows, gname = [], torch.cuda.get_device_name(dev)
    def bench(n, device, thr=None):
        if thr: torch.set_num_threads(thr)
        b = Batch.from_data_list([timing_items[i % len(timing_items)] for i in range(n)]).to(device)
        task.to(device)
        return time_fn(lambda: G.forward(task, b), device)
    for n in (1, 64, 1024):
        try: t = bench(n, dev)
        except torch.cuda.OutOfMemoryError: t = float("nan"); torch.cuda.empty_cache()
        trows.append(dict(device=gname, threads="", batch=n, sec_per_batch=t, ms_per_sample=1e3 * t / n)); print(trows[-1], flush=True)
    t = bench(1, torch.device("cpu"), 1)
    trows.append(dict(device="cpu", threads=1, batch=1, sec_per_batch=t, ms_per_sample=1e3 * t))
    pd.DataFrame(trows).to_csv(res / "timing.csv", index=False)
json.dump(dict(ckpt=f"gridfm/genco-pfdelta task_1.3/seed{a.seed}/last.pt", hf_rev="9194a31fac0bac9175ca9683893c9d8d0a477b11",
               data="gridfm/pfdelta_task1.3 rev c0988a5d0675a56a703b6654656e2aaa035941e4", graphkit=gk, host=platform.node(), torch=torch.__version__),
          open(res / "run_meta.json", "w"))
