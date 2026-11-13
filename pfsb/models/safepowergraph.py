"""Adapter for the released SafePowerGraph model zoo (Ghamizi et al., Energy and AI 2026; code
github.com/yamizi/SafePowerGraph, weights figshare project 248768, CC BY 4.0).

Nothing from upstream is vendored: the upstream clone lives in cache/safepowergraph/code (gitignored) and is put on
sys.path at run time. Upstream's own graph builder (utils.pandapower.build_dataset / PandaPowerGraph) and model class
(utils.models.GNN wrapped in torch_geometric to_hetero, as in runs/cases.py init_model) are used unchanged. Three
import-time dependencies of upstream that are not needed here (ray, oct2py/matpower, opendssdirect, comet_ml) are
replaced by empty stub modules so that the import succeeds; the stubs are never called.

Upstream facts this adapter relies on (see README, SafePowerGraph section):
  * output channels per node = [gen P, gen Q, ext_grid P, ext_grid Q, bus Vm, bus Va] (y_nodes gen, ext_grid, bus);
    bus Vm (pu) is column 4 and bus Va (radians) column 5 of the 'bus' output.
  * upstream's pandapower_pf path (opf=0) never accepts a sample, because build_one_graph tests net.OPF_converged,
    which pp.runpp does not set. `patch_runpp()` wraps pp.runpp to set it from net.converged so power flow labels can
    be generated with upstream's own builder.
"""
import copy, os, sys, types
import numpy as np
import torch

CODE = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "safepowergraph", "code")
PYLIB = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "safepowergraph", "pylib")
Y_NODES = ["gen", "ext_grid", "bus"]


def setup_upstream(code_dir=CODE):
    # upstream's weights were trained on graphs built with pandapower 2.14.11 (its requirements-exact.txt); the feature
    # layout differs under 3.x (geo column), so 2.14.11 is installed with `pip --no-deps --target` into PYLIB and put first.
    if os.path.isdir(PYLIB) and os.path.abspath(PYLIB) not in sys.path:
        assert "pandapower" not in sys.modules, "import pfsb.models.safepowergraph before pandapower"
        sys.path.insert(0, os.path.abspath(PYLIB))
        for a, b in (("Inf", "inf"), ("NaN", "nan"), ("float_", "float64"), ("product", "prod"), ("PINF", "inf")):
            if not hasattr(np, a):
                setattr(np, a, getattr(np, b))   # numpy 2 removed these aliases; pandapower 2.14 still uses them
    code_dir = os.path.abspath(code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    for name in ("ray", "comet_ml", "oct2py", "opendssdirect", "runs", "runs.matpower", "runs.opendss"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["runs.matpower"].opf = None
    sys.modules["runs.opendss"].pf = None
    sys.modules["ray"].remote = lambda f: f
    import pandapower as pp
    pp.diagnostic = lambda *a, **k: {}   # upstream only prints and stores it (slow); labels are unaffected


def patch_runpp():
    import pandapower as pp
    if getattr(pp.runpp, "_spg_patched", False):
        return
    orig = pp.runpp

    def runpp(net, *a, **k):
        orig(net, *a, **k)
        net.OPF_converged = bool(net.converged)

    runpp._spg_patched = True
    pp.runpp = runpp


def build_graphs(case, mutation, opf, n, seed, mutation_rate=0.7):
    """n labelled graphs with upstream's own builder. opf: 1 = pandapower runopp labels, 0 = runpp labels.
    Returns (graphs, networks) where networks['mutants'][i] is the solved pandapower net of graph i."""
    setup_upstream()
    if int(opf) == 0:
        patch_runpp()
    from utils.pandapower import build_dataset
    np.random.seed(seed); torch.manual_seed(seed)
    g, nets, _, _ = build_dataset(case, nbsamples=n, dataset_type="y_OPF", opf=int(opf), mutations=[mutation],
                                  mutation_rate=mutation_rate, scale=False, hetero=True, use_ray=False, device="cpu",
                                  y_nodes=Y_NODES)
    return g, nets


def graph_items(graphs):
    """Transformed HeteroData per graph, as upstream's run_train_eval does (set_device then [0])."""
    return [g.set_device("cpu", 0)[0] for g in graphs]


def hidden_from_state_dict(sd):
    ks = sorted({int(k.split(".")[1][4:]) for k in sd if k.startswith("convs.conv")})
    h = []
    for i in ks:
        w = [v for k, v in sd.items() if k.startswith(f"convs.conv{i}.bus__to__load.") and k.endswith("weight")][0]
        h.append(int(w.shape[0]))
    return h


def load_model(ckpt, item, cls, aggr="mean", device="cpu"):
    from utils.models import GNN
    from torch_geometric.nn import to_hetero
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    hidden = hidden_from_state_dict(sd)
    out = int(sd["linear2.bus.weight"].shape[0])
    model = GNN(hidden_channels=hidden, out_channels=out, aggr=aggr, cls=cls)
    model = to_hetero(model, item.metadata(), aggr="sum")
    with torch.no_grad():
        model(item.x_dict, item.edge_index_dict)
    model.load_state_dict(sd)
    return model.eval().to(device), hidden


def labels(items, slack):
    vm = np.stack([it["bus"].y[:, 0].numpy() for it in items])
    va = np.stack([np.rad2deg(it["bus"].y[:, 1].numpy()) for it in items])
    return vm, va


@torch.no_grad()
def forward(model, batch):
    return model(batch.x_dict, batch.edge_index_dict)


@torch.no_grad()
def predict(model, items, device, bs=64):
    from torch_geometric.data import Batch
    vm, va = [], []
    for k in range(0, len(items), bs):
        b = Batch.from_data_list(items[k:k + bs]).to(device)
        o = forward(model, b)["bus"]
        B = b.num_graphs
        vm.append(o[:, 4].cpu().numpy().reshape(B, -1)); va.append(np.rad2deg(o[:, 5].cpu().numpy()).reshape(B, -1))
    return np.concatenate(vm), np.concatenate(va)


def export_net(net):
    """PYPOWER-format arrays for one solved sample, in pandapower's internal (ppc) bus order, plus labels in that order.
    Gen rows hold the solved P/Q of ext_grid then gen (res_*), Vg = label Vm of the bus, so the power flow at the
    exported case has the label as its solution (checked with pfsb.export.max_mismatch)."""
    import pandapower as pp
    from pandapower.converter import to_ppc
    ppc = to_ppc(net, calculate_voltage_angles=True, init="flat", mode="pf")
    lk = net._pd2ppc_lookups["bus"]
    nb = len(ppc["bus"])
    pos = {b: i for i, b in enumerate(net.bus.index)}
    perm = np.array([lk[b] for b in net.bus.index])          # pp bus position -> ppc row
    vm = np.zeros(nb); va = np.zeros(nb)
    vm[perm] = net.res_bus.vm_pu.values; va[perm] = net.res_bus.va_degree.values
    bus = ppc["bus"][:, :13].copy(); bus[:, 0] = np.arange(nb) + 1
    gen = ppc["gen"][:, :21].copy() if ppc["gen"].shape[1] >= 21 else np.hstack([ppc["gen"], np.zeros((len(ppc["gen"]), 21 - ppc["gen"].shape[1]))])
    gp = np.concatenate([net.res_ext_grid.p_mw.values, net.res_gen.p_mw.values, net.res_sgen.p_mw.values])
    gq = np.concatenate([net.res_ext_grid.q_mvar.values, net.res_gen.q_mvar.values, net.res_sgen.q_mvar.values])
    assert len(gp) == len(gen), (len(gp), len(gen))
    gb = gen[:, 0].astype(int)
    gen[:, 0] = gb + 1; gen[:, 1] = gp; gen[:, 2] = gq; gen[:, 5] = vm[gb]
    br = ppc["branch"][:, :13].copy(); br[:, 0] += 1; br[:, 1] += 1
    return dict(bus=bus, gen=gen, branch=br, label_vm=vm, label_va_deg=va, perm=perm, baseMVA=float(ppc["baseMVA"]))
