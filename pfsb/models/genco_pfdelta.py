"""Adapter for the released GENCO PF-Delta checkpoints (gridfm/genco-pfdelta), via the gridfm-graphkit
branch genco-paper-repro-pfdelta (put on sys.path by the caller; it is a separate clone, not installed).

The branch's datamodule fits the normalizer on the train split and sets up every test subset; here the same
pieces are used directly: fit the normalizer on the train parquet files, then build one dataset per test
subset (n / n-1 / n-2 x feasible / nose) with the same transforms.
"""
import glob, os, shutil
import numpy as np
import pandas as pd
import torch
import yaml


def fit_stats(case_dir, normalizer):
    bus = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{case_dir}/train/**/bus_data.parquet", recursive=True))])
    gen = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{case_dir}/train/**/gen_data.parquet", recursive=True))])
    return normalizer.fit(bus_data=bus, gen_data=gen)


def load(ckpt, cfg_yaml, case_dir, device):
    """Returns task, stats, {subset_name: dataset}. case_dir = <pfdelta_task>/case118."""
    import gridfm_graphkit
    from gridfm_graphkit.io.param_handler import NestedNamespace, get_task, load_normalizer, get_task_transforms
    from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
    args = NestedNamespace(**yaml.safe_load(open(cfg_yaml)))
    norm = load_normalizer(args)
    stats = fit_stats(case_dir, norm)
    datasets = {}
    for tdir in sorted(glob.glob(f"{case_dir}/test/case118/*/*")):
        grid, sub = tdir.split("/")[-2:]
        proc = os.path.join(tdir, "processed"); os.makedirs(proc, exist_ok=True)
        torch.save(stats, os.path.join(proc, f"data_stats_{args.data.normalization}.pt"))
        datasets[f"case118_{grid}_{sub}"] = HeteroGridDatasetDisk(
            root=tdir, norm_method=args.data.normalization, data_normalizer=norm,
            transform=get_task_transforms(args=args), fit_normalizer=False)
    task = get_task(args, [norm])
    task.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    task.eval().to(device)
    return task, stats, datasets, gridfm_graphkit.__file__


@torch.no_grad()
def forward(task, batch):
    return task.forward(x_dict=batch.x_dict, edge_index_dict=batch.edge_index_dict,
                        edge_attr_dict=batch.edge_attr_dict, mask_dict=batch.mask_dict)


@torch.no_grad()
def predict(task, batch):
    out = forward(task, batch)["bus"]
    B = batch.num_graphs
    vm = out[:, 0].detach().cpu().numpy().reshape(B, -1)
    va = np.rad2deg(out[:, 1].detach().cpu().numpy()).reshape(B, -1)
    return batch["scenario_id"].cpu().numpy(), vm, va
