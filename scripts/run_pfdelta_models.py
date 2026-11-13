"""Evaluate a PF-Delta-recipe model (PowerFlowNet, GNS, CANOS-PF) on the fixed PF-Delta task-1.3 test subsets.

  python scripts/run_pfdelta_models.py --model {pfnet,gns,canos} --grid caseN --run-dir <PF-Delta run dir> [--root <data root>]

Writes results/models/<model>_pfdelta_<grid>/{per_sample.csv,timing.csv,id_mapping_check.json,run_meta.json} and updates
the rows of that model in results/models/summary.csv (subsets n, n-1, n-2, all). Other rows are left untouched.

Test samples: the sample ids in results/classical/test_ids_<grid>_<cond>.txt (the classical baselines' subsets) when the file
exists, otherwise the full feasible test set (last 2000 of the shuffle per condition). Each id is mapped to its position in
PF-Delta's processed test.pt (feasible n, n-1, n-2 blocks of 2000, in shuffle order) and the mapping is verified against the
raw sample JSON labels (and the cached classical npz if present) before anything is scored.

--root defaults to the root_dir in the run's config.yaml; the raw caseNN/ and shuffle_files/ must be under it.
Needs PFSB_PFDELTA_UPSTREAM (default cache/pfdelta-upstream) for the PF-Delta model code.
"""
import argparse, csv, json, os, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from pfsb.models import pfdelta_models as M
from pfsb.metrics import sample_errors
from pfsb.timing import time_fn
from pfsb.data import pfdelta as PD

CONDS = ("n", "n-1", "n-2")
BLOCK = 2000          # feasible test samples per condition; test.pt holds n, n-1, n-2 blocks first (then near-infeasible)
SUMMARY_COLS = ["vm_mae", "va_mae_deg", "vm_max", "va_max_deg", "vm_mae_pq", "va_mae_deg_pq", "vm_mae_pv", "va_mae_deg_pv"]


def read_ids(ids_dir, grid, cond):
    f = Path(ids_dir) / f"test_ids_{grid}_{cond}.txt"
    return [int(x) for x in f.read_text().split()] if f.exists() else None


def verify_mapping(grid, cond, ids, pos, ds, offset, data_root, cache_dir, n_json=25):
    """Dataset labels at the mapped positions vs raw JSON labels (evenly spaced ids) and vs the cached npz (all ids)."""
    out = {"cond": cond, "n_ids": len(ids)}
    sel = np.linspace(0, len(ids) - 1, min(n_json, len(ids))).astype(int)
    d = PD.case_dir(grid, cond, [data_root])
    worst_vm = worst_va = 0.0
    for k in sel:
        ppc = PD.sample_to_ppc(json.load(open(d / f"sample_{ids[k]}.json")))
        bv = ds[offset + pos[k]]["bus"].bus_voltages.numpy()       # [va rad, vm]
        worst_vm = max(worst_vm, np.abs(bv[:, 1] - ppc["bus"][:, 7]).max())
        worst_va = max(worst_va, np.abs(np.degrees(bv[:, 0]) - ppc["bus"][:, 8]).max())
    out.update(json_checked=len(sel), json_max_vm_diff=float(worst_vm), json_max_va_diff_deg=float(worst_va))
    npz = Path(cache_dir) / f"{grid}_{cond}_m500_s0.npz"
    if npz.exists():
        z = np.load(npz)
        if list(z["ids"]) == list(ids):
            wv = wa = 0.0
            for k in range(len(ids)):
                bv = ds[offset + pos[k]]["bus"].bus_voltages.numpy()
                wv = max(wv, np.abs(bv[:, 1] - z["bus"][k][:, 7]).max())
                wa = max(wa, np.abs(np.degrees(bv[:, 0]) - z["bus"][k][:, 8]).max())
            out.update(npz_checked=len(ids), npz_max_vm_diff=float(wv), npz_max_va_diff_deg=float(wa))
            worst_vm, worst_va = max(worst_vm, wv), max(worst_va, wa)
        else:
            out["npz"] = "ids differ from the cached npz; skipped"
    out["ok"] = bool(worst_vm < 1e-5 and worst_va < 1e-3)
    return out


def fmt(v):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return ""
    return repr(float(v)) if isinstance(v, (float, np.floating)) else str(v)


def update_summary(path, tag, df, timing):
    path = Path(path)
    with open(path, newline="") as f:
        rd = csv.DictReader(f); cols = rd.fieldnames; rows = list(rd)
    rows = [r for r in rows if r["model"] != tag]
    for name in list(CONDS) + ["all"]:
        g = df if name == "all" else df[df.condition == name]
        if not len(g):
            continue
        r = dict(model=tag, dataset=df.dataset.iloc[0], subset=name, n=len(g),
                 vm_mae=g.vm_mae.mean(), va_mae_deg=g.va_mae_deg.mean(), vm_max=g.vm_max.max(), va_max_deg=g.va_max_deg.max(),
                 vm_mae_pq=g.vm_mae_pq.mean(), va_mae_deg_pq=g.va_mae_deg_pq.mean(),
                 vm_mae_pv=g.vm_mae_pv.mean(), va_mae_deg_pv=g.va_mae_deg_pv.mean())
        if name == "all":
            r.update(timing)
        rows.append({c: fmt(r.get(c)) for c in cols})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=M.KINDS); ap.add_argument("--grid", required=True)
    ap.add_argument("--run-dir", required=True); ap.add_argument("--root", default=None)
    ap.add_argument("--ids-dir", default=str(root / "results/classical")); ap.add_argument("--cache-dir", default=str(root / "cache"))
    ap.add_argument("--out", default=None, help="result dir (default results/models/<tag>)")
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch", type=int, default=64, help="evaluation batch size"); ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--tag", default=None, help="model name in summary.csv / result dir (default <model>_pfdelta_<grid>)")
    ap.add_argument("--no-summary", action="store_true")
    a = ap.parse_args()
    tag = a.tag or f"{a.model}_pfdelta_{a.grid}"
    res = Path(a.out) if a.out else root / "results/models" / tag
    res.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else f"cuda:{a.gpu}")
    ld = M.load_run(a.model, a.run_dir, root=a.root, device=dev)
    data_root = Path(a.root or ld.dataset_cfg["root_dir"]).resolve()
    assert ld.grid == a.grid, f"run is for {ld.grid}, not {a.grid}"
    ds = ld.dataset("test", data_root)
    assert len(ds) >= 3 * BLOCK, f"test set has {len(ds)} samples"
    from torch_geometric.data import Batch
    rows, extra, checks, bench_items = [], [], [], None
    for ci, cond in enumerate(CONDS):
        full = PD.test_ids(cond, [data_root], max_samples=0)
        assert len(full) == BLOCK
        ids = read_ids(a.ids_dir, a.grid, cond) or full
        where = {s: i for i, s in enumerate(full)}
        pos = [where[s] for s in ids]                     # KeyError = an id outside the PF-Delta test split
        off = ci * BLOCK
        checks.append(verify_mapping(a.grid, cond, ids, pos, ds, off, data_root, a.cache_dir))
        print(checks[-1], flush=True)
        if not checks[-1]["ok"]:
            raise SystemExit(f"id mapping check FAILED for {cond}; nothing scored")
        items = [ds[off + p] for p in pos]
        if bench_items is None and cond == "n-1":
            bench_items = items
        VM, VA, TM, BT, SL = [], [], [], [], []
        for k in range(0, len(items), a.batch):
            chunk = items[k:k + a.batch]
            b = Batch.from_data_list(chunk)
            vb = b["bus"].bus_voltages.numpy().astype(np.float64)
            b = b.to(dev)
            vm, va = M.predict(ld, b)
            nb = len(vm) // len(chunk)
            VM.append(vm.reshape(-1, nb)); VA.append(va.reshape(-1, nb)); TM.append(vb[:, 1].reshape(-1, nb))
            TA = np.degrees(vb[:, 0]).reshape(-1, nb); SL.append(TA); BT.append(b["bus"].bus_type.cpu().numpy().reshape(-1, nb))
        VM, VA, TM, TA = (np.concatenate(x).astype(np.float64) for x in (VM, VA, TM, SL)); BT = np.concatenate(BT)
        slack = np.argmax(BT == 3, axis=1)
        err = sample_errors(VM, VA, TM, TA, slack, BT)
        rows.append(pd.DataFrame({"dataset": f"pfdelta_{a.grid}", "sample_id": ids, "condition": cond, **err}))
        # diagnostic (not part of the benchmark columns): angle MAE after removing the best constant rotation per sample
        dd = (VA - TA + 180.0) % 360.0 - 180.0
        extra.append(pd.DataFrame({"sample_id": ids, "condition": cond,
                                   "va_mae_deg_bestrot": np.abs(dd - np.median(dd, axis=1, keepdims=True)).mean(1)}))
        print(cond, len(ids), "Vm MAE %.3e pu  Va MAE %.4f deg" % (err["vm_mae"].mean(), err["va_mae_deg"].mean()), flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(res / "per_sample.csv", index=False)
    pd.concat(extra, ignore_index=True).to_csv(res / "per_sample_diag_bestrot.csv", index=False)
    json.dump(checks, open(res / "id_mapping_check.json", "w"), indent=1)
    timing = {}
    if not a.no_timing:
        trows = []
        gname = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
        def bench(n, device, thr=None):
            if thr:
                torch.set_num_threads(thr)
            ld.model.to(device)
            b = Batch.from_data_list([bench_items[i % len(bench_items)] for i in range(n)]).to(device)
            return time_fn(lambda: M.forward_only(ld, b), device)
        if dev.type == "cuda":
            for n in (1, 64, 1024):
                try:
                    t = bench(n, dev); col = n
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); t = float("nan"); col = n
                    try:
                        t = bench(512, dev); col = 512; n = 512
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                trows.append(dict(device=gname, threads="", batch=n, sec_per_batch=t, ms_per_sample=1e3 * t / n))
                timing[f"gpu_b{n}_ms_per_sample"] = 1e3 * t / n; print(trows[-1], flush=True)
        t = bench(1, torch.device("cpu"), 1)
        trows.append(dict(device="cpu", threads=1, batch=1, sec_per_batch=t, ms_per_sample=1e3 * t))
        timing["cpu1_b1_ms_per_sample"] = 1e3 * t
        pd.DataFrame(trows).to_csv(res / "timing.csv", index=False)
    json.dump(dict(model=a.model, grid=a.grid, run_dir=str(Path(a.run_dir).resolve()), host=platform.node(), torch=torch.__version__,
                   load_avg=os.getloadavg(), timing_note="GPUs shared with running training jobs; forward pass only, batches on device",
                   time=time.strftime("%Y-%m-%d %H:%M:%S")), open(res / "run_meta.json", "w"), indent=1)
    if not a.no_summary:
        update_summary(root / "results/models/summary.csv", tag, df, timing)
    print(df.groupby("condition")[["vm_mae", "va_mae_deg"]].mean())


if __name__ == "__main__":
    main()
