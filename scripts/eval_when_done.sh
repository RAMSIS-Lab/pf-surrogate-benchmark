#!/usr/bin/env bash
# Evaluate every finished PF-Delta training run under cache/pfdelta-upstream/runs/pfsb_local/ that has not been evaluated.
#
#   bash scripts/eval_when_done.sh [RUNS_DIR]        # RUNS_DIR default: cache/pfdelta-upstream/runs/pfsb_local
#   PY=<python> GPU=<index> DRY=1 bash scripts/eval_when_done.sh
#
# Run dirs are <model>_<grid>_s<seed>/<run_name>/ (model in pfnet|gns|canos). A run is finished when its summary.json says
# "finished": true or its out.txt contains "Training finished in" (PF-Delta writes both at the end), and model.pt exists.
# A run is evaluated when results/models/<model>_pfdelta_<grid>/per_sample.csv is missing or older than the run's model.pt.
# Safe to call repeatedly (or from a loop/timer); a lock keeps two invocations from running at once. Never touches training.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="${1:-$HERE/cache/pfdelta-upstream/runs/pfsb_local}"
PY="${PY:-/research/nfs_shafieezadeh_1/mahrouqi.1/conda_envs/grid-operations/bin/python}"
mkdir -p "$HERE/logs"
exec 9>"$HERE/logs/eval_when_done.lock"
flock -n 9 || { echo "another eval_when_done is running"; exit 0; }
cd "$HERE" || exit 1
pick_gpu() {   # GPU with the most free memory unless GPU is set
  if [ -n "${GPU:-}" ]; then echo "$GPU"; return; fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -n -r | head -1 | cut -d, -f1
}
n_done=0; n_skip=0; n_wait=0; n_fail=0
for d in "$RUNS"/*_s*/; do
  [ -d "$d" ] || continue
  base="$(basename "$d")"; model="${base%%_*}"; rest="${base#*_}"; grid="${rest%%_s*}"
  case "$model" in pfnet|gns|canos) ;; *) continue;; esac
  run="$(ls -dt "$d"*/ 2>/dev/null | head -1)"; run="${run%/}"
  [ -n "$run" ] || { n_wait=$((n_wait+1)); continue; }
  fin=0
  if [ -f "$run/summary.json" ] && grep -q '"finished": true' "$run/summary.json"; then fin=1; fi
  if [ "$fin" = 0 ] && [ -f "$run/out.txt" ] && grep -aq "Training finished in" "$run/out.txt"; then fin=1; fi
  if [ "$fin" = 0 ] || [ ! -f "$run/model.pt" ]; then echo "waiting  $model $grid ($(basename "$run"))"; n_wait=$((n_wait+1)); continue; fi
  out="results/models/${model}_pfdelta_${grid}"
  if [ -f "$out/per_sample.csv" ] && [ "$out/per_sample.csv" -nt "$run/model.pt" ]; then
    echo "done     $model $grid"; n_skip=$((n_skip+1)); continue; fi
  echo "evaluate $model $grid <- $run"
  [ -n "${DRY:-}" ] && continue
  log="logs/eval_${model}_${grid}.log"
  if CUDA_VISIBLE_DEVICES="$(pick_gpu)" nice -n 5 "$PY" scripts/run_pfdelta_models.py --model "$model" --grid "$grid" --run-dir "$run" > "$log" 2>&1; then
    n_done=$((n_done+1)); tail -4 "$log"
  else
    n_fail=$((n_fail+1)); echo "FAILED   $model $grid (see $log)"; tail -5 "$log"
  fi
done
echo "evaluated=$n_done already_done=$n_skip waiting=$n_wait failed=$n_fail"
