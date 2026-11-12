# pf-surrogate-benchmark

Benchmark of power-flow surrogates. This directory holds the classical-baseline half: DC, XB fast-decoupled
(FD) and Newton-Raphson solvers (numpy/scipy, single thread) evaluated on the PF-Delta task-1.3 test sets.

## Layout
- `pfsb/baselines/classical.py` solvers, timing, `run_on_ppc`, `throughput`
- `pfsb/data/pfdelta.py` PF-Delta task-1.3 test subsets as PYPOWER cases (cached as npz under `cache/`, gitignored)
- `scripts/run_classical.py` writes `results/classical/{per_sample,summary}.csv`
- `scripts/run_throughput.py` writes `results/classical/throughput.csv` (P = 1 and 16 worker processes)
- `tests/test_classical.py` Newton matches the PF-Delta label on 5 case14 samples

## Data roots
Set `PFSB_PFDELTA_ROOTS` to an `os.pathsep`-separated list of directories that contain `caseNN/{n,n-1,n-2}/raw/` and
(in one of them) `shuffle_files/`. No data is stored in this repo.

## Methods
`dc`; `dc_fd` (DC then k = 1, 2, 5, 10 FD iterations); `dc_fd_conv` (FD from DC to max mismatch 1e-8 pu, cap 50);
`nr_cold` (flat-start Newton, 1e-8, cap 30); `nr_dc` (Newton from DC). Taps and phase shifters are modelled in DC
(b = 1/(x tap), shifter injection); bus shunts are ignored in DC. FD is XB: B' = 1/x, B'' = -imag(Ybus) on PQ rows.
`t_build_ms` = matrix construction and factorisation (Ybus, DC B, FD B'/B''); `t_solve_ms` = DC solve, iterations and
convergence checks. Each method is run 3 times and the median of each time is kept. Set threads to 1 (importing
`pfsb` does it) and run with `nice`; `load_avg` is the 1-minute load at the time of the sample.
Errors are against the stored PF-Delta solution, angles after referencing both to the slack bus.
`label_mismatch` is the power mismatch of the label itself in our network (Ipopt tolerance, about 1e-6 to 4e-6 pu on
case118); `label_flag` marks > 1e-6.

## Running your own case (`run_on_ppc`)
```python
from pfsb.baselines import run_on_ppc
rows = run_on_ppc(ppc, label=None)   # list of dicts, one per (method, k)
```
`ppc` is a PYPOWER/MATPOWER-style dict with `baseMVA`, `bus`, `gen`, `branch` as numpy arrays in the standard column
order (MW/MVAr, degrees for angles and shifts, external bus numbers):
- `bus`: BUS_I, BUS_TYPE (1 PQ, 2 PV, 3 slack; exactly one slack), PD, QD, GS, BS, ..., VM (col 7), VA (col 8, deg)
- `gen`: GEN_BUS, PG, QG (the scenario's dispatch), ..., VG (col 5, the |V| setpoint for PV and slack), GEN_STATUS (col 7)
- `branch`: F_BUS, T_BUS, BR_R, BR_X, BR_B (total charging), ..., TAP (col 8, 0 = 1), SHIFT (col 9, deg), BR_STATUS (col 10)
Specified injections come from gen PG/QG and bus PD/QD; PV buses without an in-service generator are treated as PQ.
`label` is `{"vm": (n,), "va_deg": (n,)}` in bus order; by default the solved state stored in `bus` VM/VA is the label.
Returned keys: method, k, converged, iters, vm_mae, va_mae_deg, vm_max, va_max_deg, same_root, t_build_ms, t_solve_ms,
load_avg. For many scenarios, loop over them yourself (one ppc per scenario) or use `parse_ppc` + `throughput`.

## Released-weights models (pfsb/models, scripts/run_genco.py)

GENCO (gridfm, arXiv 2608.09921) IEEE case118 fine-tuned checkpoints, evaluated unchanged on the authors' own
case118_ieee test split (`gridfm_graphkit` as a dependency; `test_ratio 0.99`, `split_by_load_scenario_idx`, seed 0,
9,852 test scenarios; a fixed seed-0 subset of 2,000 is scored). Results: `results/models/<model>/per_sample.csv`,
`timing.csv`, `run_meta.json` (sample ids), and `results/models/summary.csv` (`scripts/summarize.py`).
Metrics: Vm MAE (pu) and angle MAE (deg) after referencing prediction and label to the slack bus. Timing is the
network forward pass only (batches pre-built, on device), median of 3 passes after 2 warm-ups.

### Exported scenarios for the classical solvers

Manifest: `cache/datasets/manifest.json` (gitignored; regenerate with `scripts/run_genco.py`). Each entry maps
`"<dataset>:<tag>"` to an `.npz` path. Arrays, stacked over the same samples as the model was scored on:
`bus [S,118,13]`, `gen [S,54,21]`, `branch [S,186,13]` in PYPOWER external format (columns as in PYPOWER idx_bus,
idx_gen, idx_brch), `baseMVA`, `label_vm [S,118]` (pu), `label_va_deg [S,118]` (slack at its reference angle),
`sample_ids [S]`, `condition [S]` (`branch_out=<k>`, `gen_out=<k>`, or `base`), and for convenience the model
prediction `pred_vm`, `pred_va_deg`. Bus ids are 1-based and in order, so `bus_i - 1` indexes the label arrays.
Out-of-service branches and generators keep their row with status 0. Bus Vm/Va columns are a flat start; gen Vg
holds the label Vm of its bus. Bus types come from the dataset (1 PQ, 2 PV, 3 REF, set per scenario).
Self-check: at the label voltage the power mismatch at non-slack buses is below 1.1e-7 MVA on 100 samples.

### GENCO on PF-Delta case118, task 1.3 (scripts/run_genco_pfdelta.py)

`gridfm/genco-pfdelta` (HF rev 9194a31f), `task_1.3/seed{1,2,3}/last.pt`, evaluated on `gridfm/pfdelta_task1.3`
(rev c0988a5d) `case118/test`, all six subsets (n, n-1, n-2 x feasible, nose; 2,000 / 200 graphs, none dropped), with
graphkit branch `genco-paper-repro-pfdelta` (commit 310e2f98, a separate clone put on PYTHONPATH, nothing installed).
The normalizer is fitted on the train split as the branch does (baseMVA 236.3427, vn_kv_max 345, equal to the authors'
logged stats). Reproduction check: `results/models/genco_pfdelta_t1.3_seed*/repro_check.csv` compares our PQ-bus Vm and
Va RMSE with the authors' `test_*_RMSE.csv`; they agree to about 1e-6 relative. Exports: `pfdelta_task1.3:<subset>`
entries in the manifest (one npz per subset; the `pred_*_seed1` arrays are seed-1 predictions).
Mean over seeds: `results/models/genco_pfdelta_t1.3_seed_mean.csv`.

### Not run

`gridfm/gridfm-pf-reconstruction` is not a repository on the Hugging Face Hub (model_info returns not found; the
gridfm org lists only six genco-* repos), so that model was not evaluated.

## Classical results
`results/classical/per_sample.csv`, `summary.csv` (PF-Delta task 1.3 test subsets, 500 per grid and condition; ids in
`test_ids_*.txt`), `manifest/` + `summary_manifest.csv` (exported scenarios, `scripts/run_classical_manifest.py`
scores every manifest entry not yet scored), `throughput.csv` (samples/s, P = 1 and 16; measured on a loaded shared
node, load average about 25 to 35 on 24 cores, so treat as a lower bound).
