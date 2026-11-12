"""PF-Delta (Rivera et al. 2025) task-1.3 test sets as PYPOWER-format cases, cached as one npz per (grid, condition).

Data roots come from the PFSB_PFDELTA_ROOTS environment variable (os.pathsep-separated) or the ``roots`` argument. A
root holds ``caseNN/{n,n-1,n-2}/raw/sample_*.json``; one root must also hold ``shuffle_files/`` (the split definition).

Test split, as PF-Delta's loader (core/datasets/pfdelta_dataset.py, shuffle_split_and_save_data): the file list is
``sample_{shuffle[k]+1}.json`` for k = 0..N-1 and the feasible test set is the LAST 2000 entries (task 1.3, per
condition). We take a fixed random subset of ``max_samples`` (seed 0) of those 2000 and record the sample ids.
The near-infeasible test files that the upstream loader also adds to the 1.3 test split are not used here.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TEST_SIZE = 2000


def _roots(roots=None):
    if roots is None:
        roots = os.environ.get("PFSB_PFDELTA_ROOTS", "").split(os.pathsep)
    roots = [Path(r) for r in roots if r]
    if not roots:
        raise RuntimeError("set PFSB_PFDELTA_ROOTS (os.pathsep-separated dirs holding caseNN/ and shuffle_files/)")
    return roots


def test_ids(cond: str, roots=None, max_samples: int = 500, seed: int = 0) -> list[int]:
    """Sample numbers (the NNN in sample_NNN.json) of the task-1.3 test subset for condition n / n-1 / n-2."""
    for r in _roots(roots):
        f = r / "shuffle_files" / cond / "raw_shuffle.json"
        if f.exists():
            sh = json.load(open(f))
            order = [int(sh[str(i)]) + 1 for i in range(len(sh))]
            test = order[-TEST_SIZE:]
            if max_samples and len(test) > max_samples:
                pick = np.random.default_rng(seed).choice(len(test), max_samples, replace=False)
                test = [test[i] for i in sorted(pick)]
            return test
    raise FileNotFoundError("shuffle_files not found under the data roots")


def case_dir(grid: str, cond: str, roots=None) -> Path:
    for r in _roots(roots):
        d = r / grid / cond / "raw"
        if d.is_dir():
            return d
    raise FileNotFoundError(f"{grid}/{cond}/raw not found under the data roots")


def sample_to_ppc(d: dict) -> dict:
    """One PF-Delta feasible sample (PowerModels network + Ipopt solution) -> PYPOWER dict. Bus VM/VA hold the stored
    solution (the label); gen VG holds the label |V| at the generator bus (the setpoint the surrogate task gets)."""
    net, sol = d["network"], d["solution"]["solution"]
    base = float(net["baseMVA"])
    ids = sorted(int(k) for k in net["bus"])
    pos = {b: i for i, b in enumerate(ids)}
    n = len(ids)
    pd, qd, gs, bs = (np.zeros(n) for _ in range(4))
    for l in net["load"].values():
        if l["status"]:
            pd[pos[l["load_bus"]]] += l["pd"]
            qd[pos[l["load_bus"]]] += l["qd"]
    for s in net["shunt"].values():
        if s["status"]:
            gs[pos[s["shunt_bus"]]] += s["gs"]
            bs[pos[s["shunt_bus"]]] += s["bs"]
    gens, pgq = [], np.zeros((n, 2))
    for gid in sorted(net["gen"], key=int):
        g = net["gen"][gid]
        on = g["gen_status"] == 1
        pg, qg = (sol["gen"][gid]["pg"], sol["gen"][gid]["qg"]) if on else (0.0, 0.0)
        if on:
            pgq[pos[g["gen_bus"]]] += (pg, qg)
        gens.append([g["gen_bus"], pg * base, qg * base, g["qmax"] * base, g["qmin"] * base,
                     sol["bus"][str(g["gen_bus"])]["vm"], base, int(on), g["pmax"] * base, g["pmin"] * base])
    bus = np.zeros((n, 13))
    for b in ids:
        e, s = net["bus"][str(b)], sol["bus"][str(b)]
        typ = e["bus_type"]
        assert typ in (1, 2, 3), f"bus {b} type {typ}"
        if typ == 2 and not pgq[pos[b]].any():      # as PF-Delta's loader: PV bus with no generation is PQ
            typ = 1
        bus[pos[b]] = [b, typ, pd[pos[b]] * base, qd[pos[b]] * base, gs[pos[b]] * base, bs[pos[b]] * base, e["area"],
                       s["vm"], np.degrees(s["va"]), e["base_kv"], e["zone"], e["vmax"], e["vmin"]]
    br = []
    for k in sorted(net["branch"], key=int):
        e = net["branch"][k]
        assert e["g_fr"] == 0 and e["g_to"] == 0 and abs(e["b_fr"] - e["b_to"]) < 1e-12, f"branch {k} asymmetric"
        br.append([e["f_bus"], e["t_bus"], e["br_r"], e["br_x"], e["b_fr"] + e["b_to"], e["rate_a"] * base,
                   e["rate_b"] * base, e["rate_c"] * base, e["tap"], np.degrees(e["shift"]), e["br_status"],
                   np.degrees(e["angmin"]), np.degrees(e["angmax"])])
    return {"baseMVA": base, "bus": bus, "gen": np.array(gens), "branch": np.array(br)}


def load_test_set(grid: str, cond: str, cache_dir="cache", roots=None, max_samples: int = 500, seed: int = 0):
    """(ids, [ppc, ...]) for the test subset, cached at cache_dir/{grid}_{cond}.npz (stacked ppc arrays + ids)."""
    cache = Path(cache_dir) / f"{grid}_{cond}_m{max_samples}_s{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return list(z["ids"]), [{"baseMVA": float(z["base"]), "bus": z["bus"][i], "gen": z["gen"][i],
                                 "branch": z["branch"][i]} for i in range(len(z["ids"]))]
    ids = test_ids(cond, roots, max_samples, seed)
    d = case_dir(grid, cond, roots)

    def rd(i):
        return sample_to_ppc(json.load(open(d / f"sample_{i}.json")))

    with ThreadPoolExecutor(16) as ex:
        ppcs = list(ex.map(rd, ids))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ids=np.array(ids), base=ppcs[0]["baseMVA"],
                        **{k: np.stack([p[k] for p in ppcs]) for k in ("bus", "gen", "branch")})
    return ids, ppcs
