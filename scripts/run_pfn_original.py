"""Original PowerFlowNet (Lin et al. 2024) released checkpoints on the authors' own v2 test split: accuracy, timing,
reproduction check and PYPOWER export.   python scripts/run_pfn_original.py --case 14   (14, 118, 6470rte)
Primary model row `pfn_original_<case>` = the authors' evaluation as written (their forward keeps dropout 0.2 active
at eval time; torch seed 0); `pfn_original_<case>_nodrop` = same weights with dropout off.
"""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from pfsb.models import pfn_original as P
from pfsb.metrics import sample_errors
from pfsb.timing import time_fn
from pfsb.export import max_mismatch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--ckpt", default=None); ap.add_argument("--tag", default=None)
ap.add_argument("--no-timing", action="store_true"); ap.add_argument("--no-export", action="store_true"); ap.add_argument("--no-repro", action="store_true")
a = ap.parse_args()
ckpt = a.ckpt or P.CKPT[a.case]
name = a.tag or f"pfn_original_{a.case}"
dataset = f"pfn_{a.case}"
dev = torch.device(f"cuda:{a.gpu}")
res = root / "results/models" / name; res.mkdir(parents=True, exist_ok=True)
_, _, L2, evaluate_epoch = P.upstream()

ds = P.make_dataset(a.case, "test")
n_file = np.load(P.CACHE / "data" / f"case{a.case}v2_edge_features.npy", mmap_mode="r").shape[0]
split_len = [int(n_file * r) for r in P.SPLIT]
offset = split_len[0] + split_len[1]
assert len(ds) == split_len[2]
rng = np.random.default_rng(0)
loc = np.sort(rng.choice(len(ds), a.n, replace=False)) if len(ds) > a.n else np.arange(len(ds))
gids = (offset + loc).astype(int)
print(f"{a.case}: file {n_file}, test split {len(ds)} (file rows {offset}..{n_file - 1}), scored {len(loc)}; ckpt {ckpt}", flush=True)

PV = P.pv_mask(P.raw_samples(a.case, gids)[0])

def score(net, tag):
    torch.manual_seed(0)
    p = P.predict(net, ds, loc, dev)
    sd = p["va_true"].std(0); s = int(np.argmin(sd)); assert sd[s] < 1e-3 and np.sort(sd)[1] > 1e-2   # slack: constant label angle
    bt = np.where(PV[None], 2, 1) * np.ones((len(loc), 1), int); bt[:, s] = 3
    err = sample_errors(p["vm"], p["va"], p["vm_true"], p["va_true"], np.full(len(loc), s), bt)
    d = root / "results/models" / tag; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"dataset": dataset, "sample_id": gids, "condition": "base", **err}).to_csv(d / "per_sample.csv", index=False)
    print(tag, pd.DataFrame(err).mean().round(6).to_dict(), flush=True)
    return p, s

net, ep, vl = P.load_model(ckpt, dev, dropout=None)
p, slack = score(net, name)
net0, _, _ = P.load_model(ckpt, dev, dropout=0.0)
score(net0, name + "_nodrop")

if not a.no_repro:   # authors' own loss (normalised masked L2, no regulariser) on their val and test splits
    rows = []
    for split in ("val", "test"):
        d_ = ds if split == "test" else P.make_dataset(a.case, "val")
        for mode, m in (("authors_eval_dropout_on", net), ("dropout_off", net0)):
            torch.manual_seed(0)
            l = evaluate_epoch(m, DataLoader(d_, batch_size=128, shuffle=False), L2(regularize=False), dev)
            rows.append(dict(ckpt=ckpt, split=split, n=len(d_), mode=mode, masked_l2=l, saved_val_loss=vl, saved_epoch=ep))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(res / "repro_check.csv", index=False)

if not a.no_timing:
    items = [ds[int(i)] for i in loc[:min(len(loc), 64)]]
    rows, gname = [], torch.cuda.get_device_name(dev)
    def bench(n, device, thr=None):
        if thr: torch.set_num_threads(thr)
        b = Batch.from_data_list([items[i % len(items)] for i in range(n)]).to(device); net.to(device)
        with torch.no_grad():
            return time_fn(lambda: net(b), device)
    for n in (1, 64, 1024):
        for m in [n] + [x for x in (512, 256, 128, 64, 32, 16) if x < n]:
            try: t = bench(m, dev); break
            except torch.cuda.OutOfMemoryError: t = float("nan"); torch.cuda.empty_cache()
        rows.append(dict(device=gname, threads="", batch=m, sec_per_batch=t, ms_per_sample=1e3 * t / m)); print(rows[-1], flush=True)
    t = bench(1, torch.device("cpu"), 1)
    rows.append(dict(device="cpu", threads=1, batch=1, sec_per_batch=t, ms_per_sample=1e3 * t))
    pd.DataFrame(rows).to_csv(res / "timing.csv", index=False)
    net.to(dev); torch.set_num_threads(4)

json.dump(dict(ckpt=ckpt, saved_val_loss=vl, saved_epoch=ep, upstream_code=str(P.CODE), upstream_commit_used="f31a4b7",
               n_file=n_file, n_test=len(ds), test_file_rows=[offset, n_file - 1], sample_ids=[int(i) for i in gids],
               slack_bus=slack + 1, host=platform.node(), torch=torch.__version__), open(res / "run_meta.json", "w"))

if not a.no_export:
    case, info = P.export_cases(a.case, gids)
    case["pred_vm"], case["pred_va_deg"] = p["vm"], p["va"]
    ks = list(range(0, len(gids), max(1, len(gids) // 100)))
    chk = [max_mismatch(case, k) for k in ks]
    print("export", info, "max mismatch (MVA) over", len(ks), "samples:", max(chk), flush=True)
    cd = root / "cache/datasets"; cd.mkdir(parents=True, exist_ok=True)
    pth = cd / f"{dataset}_{name}.npz"; np.savez_compressed(pth, **case)
    mp = cd / "manifest.json"; m_ = json.load(open(mp)) if mp.exists() else {}
    m_[f"{dataset}:{name}"] = dict(path=str(pth), n_samples=len(gids), baseMVA=100.0, export_selfcheck_max_mismatch_mva=max(chk))
    json.dump(m_, open(mp, "w"), indent=1)
    json.dump(dict(info=info, selfcheck_samples=len(ks), selfcheck_max_mismatch_mva=max(chk), selfcheck_median_mva=float(np.median(chk))), open(res / "export_meta.json", "w"), default=int)
