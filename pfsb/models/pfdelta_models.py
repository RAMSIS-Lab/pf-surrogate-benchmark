"""Load PF-Delta-recipe models (PowerFlowNet, GNS, CANOS-PF) from a PF-Delta run directory and predict bus voltages.

The model and dataset classes come from the pinned upstream PF-Delta clone (MOSSLab-MIT/pfdelta, commit 6fcdf2b), located by
the environment variable PFSB_PFDELTA_UPSTREAM (default: <repo>/cache/pfdelta-upstream). Models are rebuilt exactly as the
trainer built them: same registry, same config.yaml written in the run directory, weights from model.pt (the trainer
saves model.pt at each new best validation point, so it is the best checkpoint).

Conventions (output of ``predict``): Vm in per unit, Va in degrees, flat arrays over all buses of the batch in batch order.
  gns    model(data) writes data['bus'].v / .theta (radians). GNS only fixes angles up to a global rotation (its losses are
         power-balance based), so callers reference prediction and label angles to the slack bus (pfsb.metrics does).
  canos  output['bus'] = [va (rad), vm (pu)] for every bus; the PV and slack entries are the given inputs.
  pfnet  output is in normalized y space (pfnet_data_mean0_var1); de-normalized here with the stats the training
         transform used (pfnet_pfdata_stats[case]). PF-Delta's masked columns: PQ predicts (vm, va), PV predicts va,
         slack predicts none. Known entries (PV vm, slack vm and va) are taken from the sample, as the model's input.
"""
from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
KINDS = ("pfnet", "gns", "canos")


def upstream_dir() -> Path:
    return Path(os.environ.get("PFSB_PFDELTA_UPSTREAM", REPO / "cache/pfdelta-upstream")).resolve()


@contextlib.contextmanager
def _in(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


_REGISTRY_LOADED = False


def _registry():
    """Import the upstream registry (load_registry walks core/ relative to the working directory)."""
    global _REGISTRY_LOADED
    up = upstream_dir()
    if str(up) not in sys.path:
        sys.path.insert(0, str(up))
    from core.utils.registry import registry
    if not _REGISTRY_LOADED:
        from core.utils.main_utils import load_registry
        with _in(up):
            load_registry()
        _REGISTRY_LOADED = True
    return registry


@dataclass
class Loaded:
    kind: str
    grid: str
    model: torch.nn.Module
    config: dict
    dataset_cfg: dict          # constructor kwargs of the (train) dataset, with root_dir resolved
    run_dir: Path
    device: torch.device

    def dataset(self, split="test", root=None):
        """The PF-Delta dataset object (processed .pt files must exist under root; they are produced by training)."""
        reg = _registry()
        cfg = dict(self.dataset_cfg)
        name = cfg.pop("name")
        cfg["split"] = split
        if root is not None:
            cfg["root_dir"] = str(root)
        with _in(upstream_dir()):
            return reg.get_dataset_class(name)(**cfg)


def load_run(kind: str, run_dir, root=None, device="cpu") -> Loaded:
    import yaml
    assert kind in KINDS, kind
    reg = _registry()
    run_dir = Path(run_dir).resolve()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    dcfg = dict(cfg["dataset"]["datasets"][0])
    if root is not None:
        dcfg["root_dir"] = str(root)
    mcfg = dict(cfg["model"])
    mname = mcfg.pop("name")
    cls = reg.get_model_class(mname)
    if cls is None:
        raise ValueError(f"unregistered model {mname!r}")
    if mcfg.get("dataset") == "_include_":          # CANOS reads case_name / num_node_features off a dataset object
        dn = dict(dcfg); name = dn.pop("name"); dn["split"] = "test"
        with _in(upstream_dir()):
            mcfg["dataset"] = reg.get_dataset_class(name)(**dn)
    with _in(upstream_dir()):
        model = cls(**mcfg)
    ck = run_dir / "model.pt"
    if not ck.exists():
        raise FileNotFoundError(f"{ck} (run not finished or no validation point reached)")
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    dev = torch.device(device)
    model.to(dev).eval()
    return Loaded(kind, dcfg["case_name"], model, cfg, dcfg, run_dir, dev)


def _stats(grid):
    _registry()
    from core.datasets.data_stats import pfnet_pfdata_stats
    return pfnet_pfdata_stats[grid]


def _scatter_pos(batch):
    return batch["bus"].bus_type.shape[0]


@torch.no_grad()
def predict(ld: Loaded, batch):
    """(vm_pu [N], va_deg [N]) numpy arrays over all buses of ``batch`` (a collated HeteroData already on ld.device).
    For GNS the data object is modified in place by the model (fresh batches only)."""
    out = ld.model(batch)
    if ld.kind == "gns":
        vm, va = batch["bus"].v, batch["bus"].theta
    elif ld.kind == "canos":
        va, vm = out["bus"][:, 0], out["bus"][:, 1]
    else:  # pfnet
        st = _stats(ld.grid)
        ym, ys = st["mean"]["bus"]["y"].to(out.device), st["std"]["bus"]["y"].to(out.device) + 1e-7
        y = out * ys + ym                                   # columns [vm, va, pd, qd, gs, bs]
        mask = batch["bus"].x[:, -6:]                       # 1 where the model predicts that column
        vm = torch.where(mask[:, 0] > 0.5, y[:, 0], batch["bus"].bus_voltages[:, 1])
        va = torch.where(mask[:, 1] > 0.5, y[:, 1], batch["bus"].bus_voltages[:, 0])
    return vm.detach().float().cpu().numpy(), np.degrees(va.detach().float().cpu().numpy())


def forward_only(ld: Loaded, batch):
    """The timed call: forward pass only, batch already on the device."""
    with torch.no_grad():
        return ld.model(batch)
