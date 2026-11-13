"""Sanity check against PF-Delta's own metric: PF-Delta's PBL Mean (mean per-bus power-balance mismatch magnitude, computed by
upstream's universal_power_balance loss exactly as its trainer does, mean of per-batch values over batches of 64) on the
feasible test set (n, n-1, n-2 blocks, full 2000 each), next to the run's own validation PBL Mean from summary.json.

  python scripts/check_pfdelta_pbl.py --model {pfnet,gns,canos} --run-dir <run dir> [--root <data root>]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from pfsb.models import pfdelta_models as M

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=M.KINDS); ap.add_argument("--run-dir", required=True)
ap.add_argument("--root", default=None); ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
ld = M.load_run(a.model, a.run_dir, root=a.root, device=dev)
reg = M._registry()
ds = ld.dataset("test", a.root)
from core.utils import pf_losses_utils  # noqa: F401  (registers the loss)
name = {"pfnet": "PFNet", "gns": "GNS", "canos": "CANOS"}[a.model]
loss = reg.get_loss_class("universal_power_balance")(name)
from torch_geometric.loader import DataLoader
out = {}
for ci, cond in enumerate(("n", "n-1", "n-2")):
    sub = ds[ci * 2000:(ci + 1) * 2000]
    vals = []
    for b in DataLoader(sub, batch_size=64, shuffle=False):
        b = b.to(dev)
        with torch.no_grad():
            o = ld.model(b)
            vals.append(loss(o, b).item())
    out[cond] = float(np.mean(vals))
out["feasible_mean"] = float(np.mean([out["n"], out["n-1"], out["n-2"]]))
try:
    s = json.load(open(Path(a.run_dir) / "summary.json"))
    out["run_val_pbl_mean"] = s["val"][0]["PBL Mean"]
except Exception as e:
    out["run_val_pbl_mean"] = None
print(json.dumps(out))
