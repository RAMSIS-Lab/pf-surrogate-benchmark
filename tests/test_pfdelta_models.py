"""Tests for the PF-Delta-model evaluator. The first group needs no data. The second runs a real checkpoint and needs
PFSB_TEST_RUN_GNS (run dir) and PFSB_TEST_ROOT (PF-Delta data root with caseNN/ and shuffle_files/), else it is skipped."""
import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pfsb.metrics import sample_errors  # noqa: E402

spec = importlib.util.spec_from_file_location("run_pfdelta_models", ROOT / "scripts/run_pfdelta_models.py")
rpm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rpm)


def test_angles_are_slack_referenced():
    nb = 5
    vm = np.ones((2, nb)); true = np.arange(nb, dtype=float)[None].repeat(2, 0)
    pred = true + np.array([[100.0], [-40.0]])               # global rotation only (what GNS does)
    bt = np.ones((2, nb), int); bt[:, 0] = 3
    e = sample_errors(vm, pred, vm, true, np.zeros(2, int), bt)
    assert np.allclose(e["va_mae_deg"], 0) and np.allclose(e["vm_mae"], 0)


def test_summary_update_keeps_other_rows(tmp_path):
    import pandas as pd
    f = tmp_path / "summary.csv"
    f.write_text("model,dataset,subset,n,vm_mae,va_mae_deg,vm_max,va_max_deg,vm_mae_pq,va_mae_deg_pq,vm_mae_pv,va_mae_deg_pv,"
                 "gpu_b1_ms_per_sample,gpu_b64_ms_per_sample,gpu_b1024_ms_per_sample,cpu1_b1_ms_per_sample,gpu_b512_ms_per_sample\n"
                 "other,d,all,5,1e-05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,,,,,\n")
    cols = ["vm_mae", "va_mae_deg", "vm_max", "va_max_deg", "vm_mae_pq", "va_mae_deg_pq", "vm_mae_pv", "va_mae_deg_pv"]
    df = pd.DataFrame({"dataset": "pfdelta_case14", "sample_id": range(6), "condition": ["n", "n", "n-1", "n-1", "n-2", "n-2"],
                       **{c: np.arange(6, dtype=float) for c in cols}})
    for _ in range(2):                                         # idempotent: second call replaces, not duplicates
        rpm.update_summary(f, "gns_pfdelta_case14", df, {"gpu_b1_ms_per_sample": 1.5, "cpu1_b1_ms_per_sample": 9.0})
    rows = list(csv.DictReader(open(f)))
    assert [r["model"] for r in rows] == ["other", "gns_pfdelta_case14", "gns_pfdelta_case14", "gns_pfdelta_case14", "gns_pfdelta_case14"]
    assert rows[0]["vm_mae"] == "1e-05"
    allrow = [r for r in rows if r["subset"] == "all"][1]
    assert allrow["n"] == "6" and float(allrow["gpu_b1_ms_per_sample"]) == 1.5 and allrow["gpu_b64_ms_per_sample"] == ""
    assert float(allrow["vm_max"]) == 5.0 and float(allrow["vm_mae"]) == 2.5


def test_eval_when_done_dry_run(tmp_path):
    def run(d, name, summary=None, out=None):
        p = tmp_path / f"{d}/{name}"; p.mkdir(parents=True)
        (p / "model.pt").write_text("x")
        if summary:
            (p / "summary.json").write_text(summary)
        return p
    run("gns_case14_s11", "r1", '{"finished": true}')
    run("canos_case30_s11", "r2", '{"finished": false}')
    (tmp_path / "bogus_case1_s1").mkdir()
    r = subprocess.run(["bash", str(ROOT / "scripts/eval_when_done.sh"), str(tmp_path)], env={**os.environ, "DRY": "1"},
                       capture_output=True, text=True)
    assert "evaluate gns case14" in r.stdout and "waiting  canos case30" in r.stdout and "bogus" not in r.stdout


@pytest.mark.skipif(not (os.environ.get("PFSB_TEST_RUN_GNS") and os.environ.get("PFSB_TEST_ROOT")), reason="no checkpoint/data")
def test_gns_checkpoint_predict_and_mapping():
    import torch
    from torch_geometric.data import Batch
    from pfsb.models import pfdelta_models as M
    from pfsb.data import pfdelta as PD
    root = os.environ["PFSB_TEST_ROOT"]
    ld = M.load_run("gns", os.environ["PFSB_TEST_RUN_GNS"], root=root)
    ds = ld.dataset("test", root)
    full = PD.test_ids("n-1", [root], max_samples=0)
    import json
    d = PD.case_dir(ld.grid, "n-1", [root])
    for j in (0, 777, 1999):                                   # n-1 block starts at 2000 in test.pt
        ppc = PD.sample_to_ppc(json.load(open(d / f"sample_{full[j]}.json")))
        bv = ds[2000 + j]["bus"].bus_voltages.numpy()
        assert np.abs(bv[:, 1] - ppc["bus"][:, 7]).max() < 1e-5
        assert np.abs(np.degrees(bv[:, 0]) - ppc["bus"][:, 8]).max() < 1e-3
    b = Batch.from_data_list([ds[i] for i in range(3)])
    vm, va = M.predict(ld, b)
    assert vm.shape == va.shape == (b["bus"].num_nodes,) and np.isfinite(vm).all()
    assert abs(vm - b["bus"].bus_voltages[:, 1].numpy()).mean() < 0.05
