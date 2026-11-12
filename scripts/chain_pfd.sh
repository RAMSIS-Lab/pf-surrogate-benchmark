#!/bin/bash
export MLFLOW_ALLOW_FILE_STORE=true PYTHONNOUSERSITE=1 PYTHONPATH=/research/nfs_shafieezadeh_1/mahrouqi.1/repos/pf-surrogate-benchmark/cache/graphkit-pfdelta
cd /research/nfs_shafieezadeh_1/mahrouqi.1/repos/pf-surrogate-benchmark
while kill -0 3590947 2>/dev/null; do sleep 5; done
for s in 2 3; do /research/nfs_shafieezadeh_1/mahrouqi.1/conda_envs/grid-operations-genco/bin/python scripts/run_genco_pfdelta.py --seed $s --no-export > logs/pfd_seed$s.log 2>&1; done
/research/nfs_shafieezadeh_1/mahrouqi.1/conda_envs/grid-operations-genco/bin/python scripts/run_genco_pfdelta.py --seed 1 --no-export --no-timing > logs/pfd_seed1b.log 2>&1
