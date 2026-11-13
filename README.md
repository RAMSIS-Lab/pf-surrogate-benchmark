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

### GENCO datakit checkpoints, base size, seed0 (scripts/run_genco_datakit.py)

`gridfm/genco-pf-datakit` (HF rev f5eb20ee) `<case>/base/seed0/best_model_state_dict.pt` with the checkpoint's own yaml and
`normalizer_stats.pt`, on IEEE 14, 30, 57, 118 and GOC 500, each on the test split of its dataset `gridfm/pf_small_<case>`
(graphkit's split: yaml seed, test_ratio 0.1 by load-scenario id, reproduced by `pfsb/models/genco_datakit.py`; about 22,000
test scenarios per grid, a seed-0 subset of 2,000 is scored). Run with the installed graphkit 0.0.7 (branch
`genco-paper-repro`, commit 95b90eec). Graphs are built only for the scored scenarios with graphkit's own builder,
normalizer and transforms. Exports: `<case>:genco_datakit_<case>_base_seed0` in the manifest.

### PF-Delta recipe models: PowerFlowNet, GNS, CANOS-PF (scripts/run_pfdelta_models.py)

Trained with PF-Delta's own code and task-1.3 recipe (pinned upstream clone, commit 6fcdf2b; configs in `scripts/train_upstream`). `pfsb/models/pfdelta_models.py` rebuilds a model from a run directory (`config.yaml` + `model.pt`)
through upstream's registry and returns Vm (pu) and Va (deg) per bus. Notes: GNS fixes angles only up to a global rotation, so
prediction and label angles are both referenced to the slack bus (as for every model here); `per_sample_diag_bestrot.csv`
adds the angle MAE after removing the best constant rotation per sample (diagnostic only). CANOS reads its bus output;
PowerFlowNet's normalized output is de-normalized with the training statistics and the known inputs (PV Vm, slack Vm and Va)
are taken from the sample. Vm at PV buses and Vm and Va at the slack bus are given inputs, so they count as zero error.

    python scripts/run_pfdelta_models.py --model {pfnet,gns,canos} --grid case118 --run-dir <run dir> [--root <data root>]
    bash scripts/eval_when_done.sh          # evaluates every finished, not yet evaluated run under cache/pfdelta-upstream/runs/pfsb_local
    python scripts/check_pfdelta_pbl.py --model gns --run-dir <run dir>   # PF-Delta's own PBL Mean on the test set

Test samples are the ids in `results/classical/test_ids_<grid>_<cond>.txt`. Each id is mapped to its row in PF-Delta's
processed `test.pt` (feasible n, n-1, n-2 blocks of 2000 in shuffle order) and the labels at the mapped rows are checked against
the raw sample JSON and the cached classical npz before scoring (`id_mapping_check.json`); a mismatch aborts the run.
Timing: forward pass only, batches on device, median of 3 after warm-up, GPU batch 1, 64, 1024 and CPU 1 thread batch 1; the
GPUs and CPUs are shared with training jobs, so treat as an upper bound. Rows `*_aug` are the August 2026 case118 GNS and
CANOS runs (same recipe, trained on OSC). Tests: `pytest tests/test_pfdelta_models.py`.

### Original PowerFlowNet release (scripts/run_pfn_original.py)

The authors' released checkpoints (SURFdrive iunfVTGsABT5NaD) on their released v2 data (SURFdrive Qw4RHLvI2RPBIBL), cases 14, 118, 6470rte.
Upstream `StavrosOrf/PoweFlowNet` is a pinned dependency, not vendored: `git archive f31a4b7` (2023-06-28, the revision that matches the
June 2023 checkpoints; repo HEAD is 1ebf73b, whose data format and model input differ) into `cache/pfn_original/code_f31a4b7`
(`PFSB_PFN_CODE`). `pfsb/models/pfn_original.py` uses their `PowerFlowData` (features, per-split normalisation), `MaskEmbdMultiMPN`,
`Masked_L2_loss`, `evaluate_epoch`. Test split = last 30 % of each file (their `[.5, .2, .3]`); 2,000-sample seed-0 subset, ids in `run_meta.json`.
Checkpoints (MaskEmbdMultiMPN, mse_loss; best val loss in `save_logs.json` among runs of that case): 14 -> `model_20230627-576` (val 0.0026);
118 -> `model_20230627-9288` (standard config, val 0.0220, the paper's Table I value; the large-config run `model_20230628-6515`, val 0.0029,
is reported as `pfn_original_118_large`); 6470rte -> `model_20230627-1251` (val 0.2904; paper 0.303).
Deviations forced by upstream at that commit: (1) its processed-file name ignores the split, so we subclass to keep splits apart;
(2) its `forward` creates `nn.Dropout(0.2)` on the fly, so dropout is ACTIVE at eval (fixed upstream in 2024). The primary rows
`pfn_original_<case>` are the authors' evaluation as written (dropout on, torch seed 0, which reproduces their val loss); `*_nodrop` is the same
weights with dropout off. Inputs the authors' mask marks as known (PV Vm, slack Va when it is 0) are taken from the sample (zero error); the slack
Vm is NOT an input in these files, the model predicts it. Normalisation statistics are those of the evaluated split (as upstream does).
`repro_check.csv` holds their masked L2 on the full val and test splits against `save_logs.json`.
Export: line r, x per sample come from the edge file; transformer x = edge r column / 990, other transformer data, shunts and the generator list
from the pandapower base case; 4 case118 transformer-like rows (conversion artefact) get r re-estimated from the labels. See `export_cases`
docstring and `export_meta.json` (max power mismatch at the label over 100 samples, MVA). The classical solvers are given the slack Vm.

### Not run

`gridfm/gridfm-pf-reconstruction` is not a repository on the Hugging Face Hub (model_info returns not found; the
gridfm org lists only six genco-* repos), so that model was not evaluated.

## Classical results
`results/classical/per_sample.csv`, `summary.csv` (PF-Delta task 1.3 test subsets, 500 per grid and condition; ids in
`test_ids_*.txt`), `manifest/` + `summary_manifest.csv` (exported scenarios, `scripts/run_classical_manifest.py`
scores every manifest entry not yet scored), `throughput.csv` (samples/s, P = 1 and 16; measured on a loaded shared
node, load average about 25 to 35 on 24 cores, so treat as a lower bound).
