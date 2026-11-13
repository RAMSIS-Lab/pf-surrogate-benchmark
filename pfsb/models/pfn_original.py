"""Adapter for the ORIGINAL PowerFlowNet release (Lin et al., IJEPES 2024; github.com/StavrosOrf/PoweFlowNet, MIT).

The authors' code is a pinned dependency, not vendored: it is checked out (git archive of commit f31a4b7, the
2023-06-28 revision that matches the released June 2023 checkpoints: 6 node features, one-hot type + mask input
of 16 columns, per-split normalisation) into PFSB_PFN_CODE (default cache/pfn_original/code_f31a4b7).
We use their PowerFlowData class (feature construction and normalisation), MaskEmbdMultiMPN, Masked_L2_loss and
evaluate_epoch. Two deviations, both forced by the upstream code at that commit (see README):
  1. `processed_file_names` there does not depend on the split, so train, val and test would all read the train
     processing; we subclass it so that each split gets its own processed file (the split itself is theirs:
     first 50 %, next 20 %, last 30 % of the file, torch.split with int(N * ratio)).
  2. `forward` builds `nn.Dropout(p)(x)` on the fly, so dropout stays active under model.eval(); the authors
     fixed that in 2024 (da17b7c). `set_dropout(model, 0)` gives the intended deterministic evaluation.
Raw files are read with numpy only (allow_pickle off); checkpoints are read with weights_only=True.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache/pfn_original"
CODE = Path(os.environ.get("PFSB_PFN_CODE", CACHE / "code_f31a4b7"))
SPLIT = [0.5, 0.2, 0.3]
# checkpoint choice, per case: best val loss among the authors' MaskEmbdMultiMPN / mse_loss runs trained on that case
CKPT = {
    "14": "model_20230627-576.pt",
    "118": "model_20230627-9288.pt",       # standard config, val 0.0220 = the paper's Table I value (0.022)
    "6470rte": "model_20230627-1251.pt",
}
ALT_CKPT = {"118": ["model_20230628-6515.pt"]}   # large config (hidden 512, 5 layers), val 0.0029, run as pfn_original_118_large
CFG = {  # hidden, layers from the authors' configs (standard / large); output 6, K 3, dropout 0.2, edge dim 5
    "model_20230627-576.pt": dict(hidden_dim=129, n_gnn_layers=4),
    "model_20230627-9288.pt": dict(hidden_dim=129, n_gnn_layers=4),
    "model_20230627-1251.pt": dict(hidden_dim=129, n_gnn_layers=4),
    "model_20230628-6515.pt": dict(hidden_dim=512, n_gnn_layers=5),
}


def upstream():
    """Import the authors' modules from the pinned checkout."""
    if not CODE.exists():
        raise FileNotFoundError(f"{CODE} missing: git archive f31a4b7 of StavrosOrf/PoweFlowNet into it")
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    try:
        import torchvision  # noqa: F401  (upstream's loss module imports it but never uses it)
    except ImportError:
        import types
        tv = types.ModuleType("torchvision"); tv.datasets = types.ModuleType("torchvision.datasets")
        tv.transforms = types.ModuleType("torchvision.transforms"); sys.modules.update(
            {"torchvision": tv, "torchvision.datasets": tv.datasets, "torchvision.transforms": tv.transforms})
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])   # our own processed files
    from datasets.PowerFlowData import PowerFlowData
    from networks.MPN import MaskEmbdMultiMPN
    from utils.custom_loss_functions import Masked_L2_loss
    from utils.evaluation import evaluate_epoch
    return PowerFlowData, MaskEmbdMultiMPN, Masked_L2_loss, evaluate_epoch


def make_dataset(case, task):
    """The authors' PowerFlowData for `task` in train/val/test of grid `case` ('14', '118', '6470rte')."""
    PFD = upstream()[0]
    root = CACHE / "proc" / case
    raw = root / "raw"; raw.mkdir(parents=True, exist_ok=True)
    for name in ("adjacency_matrix", "edge_features", "node_features_x", "node_features_y"):
        src = CACHE / "data" / f"case{case}v2_{name}.npy"; dst = raw / src.name
        if not dst.exists():
            dst.symlink_to(src)

    class TaskPFD(PFD):
        @property
        def processed_file_names(self):
            return [f"case{self.case}_processed_{self.task}.pt"]
    return TaskPFD(root=str(root), case=f"{case}v2", split=SPLIT, task=task, normalize=True)


def load_model(ckpt, device, dropout=0.0):
    """Authors' MaskEmbdMultiMPN with the released weights. dropout=None keeps their (active-at-eval) dropout 0.2."""
    _, Net, _, _ = upstream()
    torch.serialization.add_safe_globals([argparse.Namespace])
    d = torch.load(CACHE / "weights" / ckpt, map_location="cpu", weights_only=True)
    c = CFG[ckpt]
    net = Net(nfeature_dim=6, efeature_dim=5, output_dim=6, hidden_dim=c["hidden_dim"], n_gnn_layers=c["n_gnn_layers"],
              K=3, dropout_rate=0.2)
    net.load_state_dict(d["model_state_dict"])
    net.eval().to(device)
    if dropout is not None:
        net.dropout_rate = dropout   # forward() builds nn.Dropout(self.dropout_rate) per call
    return net, d["epoch"], float(d["val_loss"])


def denorm(ds, z):
    mean, std = ds.xymean[0].to(z.device), ds.xystd[0].to(z.device)
    return z * (std + 1e-7) + mean


@torch.no_grad()
def predict(net, ds, idx, device, bs=64):
    """Per sample in `idx`: Vm, Va(deg) prediction and label, bus type (1 PQ, 2 PV, 3 slack), inputs.
    Known inputs (where the authors' mask is 0) are taken from the sample, so they carry zero error."""
    from torch_geometric.data import Batch
    out = dict(vm=[], va=[], vm_true=[], va_true=[], p_true=[], q_true=[], ftype=[])
    for k in range(0, len(idx), bs):
        items = [ds[int(i)] for i in idx[k:k + bs]]
        b = Batch.from_data_list(items).to(device)
        o = net(b)
        mask = b.x[:, 10:]
        z = torch.where(mask > 0, o, b.x[:, 4:10])
        z, y = denorm(ds, z), denorm(ds, b.y)
        n = len(items); nb = z.shape[0] // n
        out["vm"].append(z[:, 0].reshape(n, nb).cpu()); out["va"].append(z[:, 1].reshape(n, nb).cpu())
        out["vm_true"].append(y[:, 0].reshape(n, nb).cpu()); out["va_true"].append(y[:, 1].reshape(n, nb).cpu())
        out["p_true"].append(y[:, 2].reshape(n, nb).cpu()); out["q_true"].append(y[:, 3].reshape(n, nb).cpu())
        out["ftype"].append(b.x[:, :4].argmax(1).reshape(n, nb).cpu())
    return {k: torch.cat(v).numpy().astype(np.float64) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------------------------
# Export of the scored samples as PYPOWER cases
# ---------------------------------------------------------------------------------------------------------------
def pv_mask(X):
    """PV buses: typed 1 in the file, or (case6470rte types every bus 2) carrying a varying Vm setpoint in the input."""
    return (X[0, :, 1] == 1) | (X[:, :, 2].std(0) > 1e-9)


def raw_samples(case, gids):
    """Raw (x, y, edge) rows of the shared v2 files for global sample ids (numpy, allow_pickle off)."""
    out = []
    for name in ("node_features_x", "node_features_y", "edge_features"):
        a = np.load(CACHE / "data" / f"case{case}v2_{name}.npy", mmap_mode="r")
        out.append(np.array(a[np.asarray(gids)], dtype=np.float64))
    return out


def base_ppc(case):
    """PYPOWER view of the pandapower base case the authors' generator started from (pandapower.networks.case<N>)."""
    import pandapower.networks as nw
    from pandapower.converter.pypower.to_ppc import to_ppc
    net = getattr(nw, "case" + case)()
    ppc = to_ppc(net, init="flat")
    assert (ppc["bus"][:, 0] == np.arange(len(ppc["bus"]))).all(), "pandapower bus order differs from the data's"
    return net, ppc


def _ybus(branch, nb):
    """Sparse branch admittance matrix (PYPOWER pi model, external 1-based bus numbers, tap 0 = 1)."""
    import scipy.sparse as sp
    f = branch[:, 0].astype(int) - 1; t = branch[:, 1].astype(int) - 1
    ys = 1 / (branch[:, 2] + 1j * branch[:, 3]); tap = np.where(branch[:, 8] == 0, 1, branch[:, 8]) * np.exp(1j * np.deg2rad(branch[:, 9]))
    bc = 1j * branch[:, 4] / 2
    rows = np.r_[f, f, t, t]; cols = np.r_[f, t, f, t]
    vals = np.r_[(ys + bc) / (tap * np.conj(tap)), -ys / np.conj(tap), -ys / tap, ys + bc]
    return sp.coo_matrix((vals, (rows, cols)), shape=(nb, nb)).tocsr()


def _calibrate_r(case_d, rows, Y, X, slack, nsamp=8):
    """Least-squares r of the transformer-like rows `rows` (x, b, tap fixed) from the labelled power balance."""
    from scipy.optimize import least_squares
    nb = case_d["bus"].shape[1]
    V = [case_d["label_vm"][k] * np.exp(1j * np.deg2rad(case_d["label_va_deg"][k])) for k in range(nsamp)]
    Sinj = [-(Y[k, :, 4] + 1j * Y[k, :, 5]) for k in range(nsamp)]   # net consumption in the labels includes shunts
    def resid(r):
        out = []
        for k in range(nsamp):
            br = case_d["branch"][k].copy(); br[rows, 2] = r
            d = V[k] * np.conj(_ybus(br, nb) @ V[k]) - Sinj[k]; d[slack] = 0
            out += [d.real, d.imag]
        return np.concatenate(out)
    r0 = case_d["branch"][0, rows, 2].copy()
    sol = least_squares(resid, r0, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=50)
    return sol.x, float(np.abs(sol.fun).max() * float(case_d["baseMVA"]))


def export_cases(case, gids, base_mva=100.0):
    """Rebuild each sample as a PYPOWER case. The shared files hold per-sample line r, x (pu), loads, generator
    P and Vm, and the solved state, but NOT the full transformer data (the edge file's r column holds 990 * x of the
    transformer, x and b hold 0) or the generator list, so those come from the pandapower base case
    (pandapower.networks.case<N>, the generator's starting point):
      lines        : per-sample r, x from the edge file, b = 0, tap 1 (the labels satisfy the power balance with b = 0);
      transformers : x = (edge r column) / 990; r, tap, shift from the base case rows (same order as the edge file),
                     b = 0. Rows whose base x differs from x_edge (a base-case conversion artefact, 4 rows in case118)
                     get tap 1 and r re-estimated by least squares from the labelled power balance of 8 samples
                     (the fit reaches 1e-10 MVA there);
      buses        : Pd, Qd = x[:, 4:6] * baseMVA (PQ buses where it disagrees with the label net consumption, 6470rte's static generators: the label value); type from the file (1 = PV, 2 = PQ; the slack bus is the one whose
                     label angle is constant over samples and is typed 2 like the PQ buses in the file);
      shunts       : base-case bus shunts (GS, BS) are included iff that lowers the label power mismatch;
      generators   : one per PV bus plus the slack; PG, QG = Pd, Qd minus the label net consumption y[:, 4:6], corrected
                     for the shunt term; VG = label Vm of its bus (the slack Vm is not an input of the PowerFlowNet files).
    """
    from pfsb.export import max_mismatch
    X, Y, E = raw_samples(case, gids)
    S, nb = X.shape[0], X.shape[1]
    net, ppc = base_ppc(case)
    nl, nt = len(net.line), len(net.trafo)
    br0 = ppc["branch"]
    assert E.shape[1] == nl + nt == len(br0)
    assert np.array_equal(E[0, :, 0:2].astype(int) - 1, br0[:, 0:2].astype(int)), "branch order differs from base case"
    assert (E[:, :, 0:2] == E[0, :, 0:2]).all() and (E[:, nl:, 2] > 0).all()
    ftype = X[:, :, 1].astype(int)
    assert (ftype == ftype[0]).all(), "bus types vary across samples"
    slack = np.where(Y[:, :, 3].std(0) < 1e-9)[0]   # the slack angle is the same in every sample (0 deg, or 30 deg for case118)
    assert len(slack) == 1, f"slack bus not unique: {slack}"
    slack = int(slack[0])
    assert slack == int(net.ext_grid.bus.values[0]), "slack differs from the base case's ext_grid bus"
    pvm = pv_mask(X)
    btype = np.where(pvm, 2, 1); btype[slack] = 3
    gbus = np.where(btype >= 2)[0]
    ng = len(gbus)
    sh = ppc["bus"][:, 4:6]
    bus = np.zeros((S, nb, 13)); gen = np.zeros((S, ng, 21)); br = np.zeros((S, nl + nt, 13))
    bus[:, :, 0] = np.arange(1, nb + 1); bus[:, :, 1] = btype
    bus[:, :, 2] = X[:, :, 4] * base_mva; bus[:, :, 3] = X[:, :, 5] * base_mva
    bus[:, :, 6] = 1; bus[:, :, 7] = 1.0; bus[:, :, 8] = 0.0; bus[:, :, 9] = ppc["bus"][:, 9]; bus[:, :, 10] = 1
    bus[:, :, 11] = 1.1; bus[:, :, 12] = 0.9
    gen[:, :, 0] = gbus + 1; gen[:, :, 3] = 9999; gen[:, :, 4] = -9999; gen[:, :, 5] = Y[:, gbus, 2]
    gen[:, :, 6] = base_mva; gen[:, :, 7] = 1; gen[:, :, 8] = 9999; gen[:, :, 9] = -9999
    br[:, :, 0:2] = E[:, :, 0:2]
    br[:, :nl, 2] = E[:, :nl, 2]; br[:, :nl, 3] = E[:, :nl, 3]
    br[:, nl:, 2] = br0[nl:, 2]; br[:, nl:, 8] = br0[nl:, 8]; br[:, nl:, 9] = br0[nl:, 9]
    if np.median(np.abs(br0[nl:, 3] - E[0, nl:, 2] / 990.0) / br0[nl:, 3]) < 0.05:   # case14, case118: transformer x is stored as r / 990
        xt = E[0, nl:, 2] / 990.0
        br[:, nl:, 3] = xt[None]
        bad = np.where(np.abs(br0[nl:, 3] - xt) > 1e-7)[0] + nl
    else:                               # case6470rte: the edge file's transformer columns are not usable; base rows
        br[:, nl:, 3] = br0[nl:, 3][None]
        bad = np.array([], dtype=int)
    br[:, :, 10] = 1; br[:, :, 11] = -360; br[:, :, 12] = 360
    case_d = dict(bus=bus, gen=gen, branch=br, label_vm=Y[:, :, 2].copy(), label_va_deg=Y[:, :, 3].copy(),
                  baseMVA=np.array(base_mva), sample_ids=np.asarray(gids), condition=np.array(["base"] * S))
    info = dict(slack_bus=slack + 1, n_gen=ng, n_lines=nl, n_trafos=nt, recalibrated_rows=[int(r) for r in bad])
    if len(bad):
        br[:, bad, 8] = 0; br[:, bad, 9] = 0
        r, res = _calibrate_r(case_d, bad, Y, X, slack)
        br[:, bad, 2] = r[None]
        info["recalibrated_r"] = [float(v) for v in r]; info["recalibration_max_resid_mva"] = res
    def set_gen(use_shunt):
        bus[:, :, 4:6] = sh[None] if use_shunt else 0.0
        vm2 = case_d["label_vm"] ** 2
        bus[:, :, 2] = X[:, :, 4] * base_mva; bus[:, :, 3] = X[:, :, 5] * base_mva
        pg = (X[:, :, 4] - Y[:, :, 4]) * base_mva + bus[:, :, 4] * vm2
        qg = (X[:, :, 5] - Y[:, :, 5]) * base_mva - bus[:, :, 5] * vm2
        gen[:, :, 1] = pg[:, gbus]; gen[:, :, 2] = qg[:, gbus]
        # PQ buses: the label net consumption is the specified injection (case6470rte has static generators there,
        # i.e. negative loads that the input features do not carry)
        pq = btype == 1
        pdy = Y[:, :, 4] * base_mva - sh[None, :, 0] * vm2; qdy = Y[:, :, 5] * base_mva + sh[None, :, 1] * vm2   # base shunts assumed in the label
        diff = pq[None] & ((np.abs(pdy - bus[:, :, 2]) > 1e-4) | (np.abs(qdy - bus[:, :, 3]) > 1e-4))
        bus[:, :, 2] = np.where(diff, pdy, bus[:, :, 2]); bus[:, :, 3] = np.where(diff, qdy, bus[:, :, 3])
    m = {}
    for use in ((False, True) if np.abs(sh).max() > 0 else (False,)):
        set_gen(use); m[use] = sum(max_mismatch(case_d, k) for k in range(min(3, S)))
    use = len(m) == 2 and m[True] < m[False]
    set_gen(use)
    info.update(shunt_used=bool(use), mismatch_noshunt_3samples=m[False])
    return case_d, info
