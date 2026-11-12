"""Adapter for the released GENCO datakit checkpoints (gridfm/genco-pf-datakit, base size, seed0).

Reproduces graphkit's own scenario split (shuffle with the yaml seed, split by load-scenario id with the yaml
ratios, via graphkit's split function) from the raw parquet files, then builds graphs only for the requested
test scenarios (graphkit's own graph builder, normalizer stats and transforms), instead of processing all
scenarios of the dataset to disk.
"""
import glob, os, random
import numpy as np
import pandas as pd
import torch
import yaml


def _part_dirs(raw, table):
    ds = sorted(glob.glob(f"{raw}/{table}_data.parquet/scenario_partition=*"), key=lambda p: int(p.split("=")[-1]))
    return ds


def test_scenario_ids(raw, cfg):
    """Scenario ids of graphkit's test split. Returns (ids, n_total)."""
    from torch.utils.data import Subset
    from gridfm_graphkit.datasets.utils import split_dataset_by_load_scenario_idx
    chunks = []
    for d in _part_dirs(raw, "bus"):
        b = pd.read_parquet(d, columns=["scenario", "load_scenario_idx"])
        chunks.append(b.groupby("scenario", sort=True)["load_scenario_idx"].first())
    ls = pd.concat(chunks).sort_index()
    assert (ls.index.values == np.arange(len(ls))).all()
    load_scenarios = torch.tensor(ls.values)
    n = len(ls)
    num = min(int(cfg["data"]["scenarios"][0]), n)
    idx = list(range(n)); random.seed(cfg["seed"]); random.shuffle(idx)
    subset = idx[:num]
    np.random.seed(cfg["seed"])
    _, _, test = split_dataset_by_load_scenario_idx(Subset(list(range(num)), list(range(num))), "", load_scenarios[subset],
                                                    cfg["data"]["val_ratio"], cfg["data"]["test_ratio"])
    ids = [subset[i] for i in test.indices]
    return ids, n


def read_scenarios(raw, ids):
    """DataFrames (bus, gen, branch) restricted to the scenario ids."""
    ids = set(int(i) for i in ids)
    out = []
    for t in ("bus", "gen", "branch"):
        parts = []
        for d in _part_dirs(raw, t):
            df = pd.read_parquet(d)
            m = df.scenario.isin(ids)
            if m.any():
                parts.append(df[m])
        out.append(pd.concat(parts))
    return tuple(out)


def load_model(ckpt_dir, network, device):
    from gridfm_graphkit.io.param_handler import NestedNamespace, get_task, load_normalizer, get_task_transforms
    from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model
    name = [f for f in os.listdir(ckpt_dir) if f.endswith(".yaml")][0]
    cfg = yaml.safe_load(open(os.path.join(ckpt_dir, name)))
    args = NestedNamespace(**cfg)
    norm = load_normalizer(args=args)
    stats = torch.load(os.path.join(ckpt_dir, "normalizer_stats.pt"), map_location="cpu", weights_only=True)
    norm.fit_from_dict(stats[network])
    task = get_task(args, [norm])
    sd = torch.load(os.path.join(ckpt_dir, "best_model_state_dict.pt"), map_location="cpu")
    task.load_state_dict(adapt_state_dict_for_model(sd, task))
    task.eval().to(device)
    return cfg, args, norm, task


def build_items(args, norm, tabs, ids):
    from gridfm_graphkit.datasets.hetero_preprocess import build_hetero_data_for_scenario, merge_agg_gen_into_bus
    from gridfm_graphkit.io.param_handler import get_task_transforms
    from torch_geometric.data import HeteroData
    bus, gen, br = tabs
    bus = merge_agg_gen_into_bus(bus, gen)
    bg, gg, rg = bus.groupby("scenario"), gen.groupby("scenario"), br.groupby("scenario")
    tf = get_task_transforms(args=args)
    items = []
    for s in ids:
        d = build_hetero_data_for_scenario(int(s), bg.get_group(s), gg.get_group(s), rg.get_group(s))
        d = HeteroData.from_dict(d.to_dict())
        norm.transform(data=d)
        items.append(tf(d))
    return items


@torch.no_grad()
def forward(task, batch):
    return task.forward(x_dict=batch.x_dict, edge_index_dict=batch.edge_index_dict,
                        edge_attr_dict=batch.edge_attr_dict, mask_dict=batch.mask_dict)


@torch.no_grad()
def predict(task, batch):
    out = forward(task, batch)["bus"]
    B = batch.num_graphs
    return (batch["scenario_id"].cpu().numpy(), out[:, 0].detach().cpu().numpy().reshape(B, -1),
            np.rad2deg(out[:, 1].detach().cpu().numpy()).reshape(B, -1))
