"""Adapter for the released GENCO (gridFM) PF checkpoints, via gridfm-graphkit as a dependency.

Uses the checkpoint's own config with the authors' evaluation split (test_ratio 0.99, val_ratio 0,
split_by_load_scenario_idx, seed 0), their normalizer stats, and graphkit's datamodule and task classes.
Model space: Va is in radians, Vm in p.u.; both are returned in p.u. / degrees here.
"""
import copy, os
import numpy as np
import torch
import yaml


def load(ckpt_dir, data_path, network, device, eval_overrides=True, workers=0):
    from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
    from gridfm_graphkit.io.param_handler import NestedNamespace, get_task
    from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model
    name = os.path.basename(ckpt_dir.rstrip("/"))
    cfg = yaml.safe_load(open(os.path.join(ckpt_dir, name + ".yaml")))
    if eval_overrides:  # the authors' case118_eval.yaml differs from the training yaml only here
        cfg["data"].update(test_ratio=0.99, val_ratio=0, workers=workers)
        cfg["data"]["scenarios"] = [250000]
    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, data_path, normalizer_stats_path=os.path.join(ckpt_dir, "normalizer_stats.pt"))
    dm.setup("test")
    task = get_task(args, dm.data_normalizers)
    sd = torch.load(os.path.join(ckpt_dir, "best_model_state_dict.pt"), map_location="cpu")
    task.load_state_dict(adapt_state_dict_for_model(sd, task))
    task.eval().to(device)
    return dm, task


def to_dev(batch, device):
    return batch.to(device)


@torch.no_grad()
def forward(task, batch):
    return task.forward(x_dict=batch.x_dict, edge_index_dict=batch.edge_index_dict,
                        edge_attr_dict=batch.edge_attr_dict, mask_dict=batch.mask_dict)


@torch.no_grad()
def predict(task, batch):
    """Returns per-graph lists: scenario ids [B], vm [B,nb], va_deg [B,nb] (equal-size graphs)."""
    out = forward(task, batch)["bus"]
    B = batch.num_graphs
    vm = out[:, 0].detach().cpu().numpy().reshape(B, -1)
    va = np.rad2deg(out[:, 1].detach().cpu().numpy()).reshape(B, -1)
    return batch["scenario_id"].cpu().numpy(), vm, va
