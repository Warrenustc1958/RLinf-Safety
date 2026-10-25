#!/usr/bin/env bash
# fresh4 step 5: one-shot collection worker (gpu006 allows one LIBERO launcher at a time; the caller
# must hold /tmp/rlinf-libero-run.lock, see chain_fresh4.sh / retry_fresh4.sh).
# One run per cell: 40 envs = the 40 fresh init states, widened program, up to 3 branches per root.
# Cells with EXIT=0 are skipped. Stops (exit 6) when /data_cf free space < MIN_FREE_GB.
#   GPUS=6-7 ROLLOUT_GPUS=5 MIN_FREE_GB=40 bash collect_fresh4.sh
set -uo pipefail
O=${FRESH4:-/data_cf/cws_sam/v1/fresh4}
ROOT=${FRESH4_CF:-/data_cf/cws/fresh4_cf}
INIT_ROOT=${INIT_ROOT:-/data_cf/cws/fresh_init/v2}
GPUS=${GPUS:-6-7}; ROLLOUT_GPUS=${ROLLOUT_GPUS:-5}; MIN_FREE_GB=${MIN_FREE_GB:-40}
TRIAL=/mnt/nfs/fjj/Code/CodeWorld-Safe/toolkits/codeworld_safe/cf/trial_step_e.sh
mapfile -t CELLS < <(python3 -c "import json; print('\n'.join(sorted(json.load(open('$O/split_fresh4.json'))['programs'])))")
export GPU_WAIT_DISABLE=1 RLINF_LIBERO_RUN_LOCK_HELD=1
for cell in "${CELLS[@]}"; do
  out=$ROOT/runs/$cell
  if grep -q "EXIT=0" "$out/launcher.log" 2>/dev/null; then echo "[skip] $cell"; continue; fi
  free=$(df --output=avail -BG /data_cf | tail -1 | tr -dc 0-9)
  if [ "$free" -lt "$MIN_FREE_GB" ]; then echo "[stop] /data_cf free ${free}G < ${MIN_FREE_GB}G"; exit 6; fi
  suite=${cell%_L*}; rest=${cell##*_L}; level=L${rest%%_t*}; task=${rest##*_t}
  prog=$(python3 -c "import json; print(json.load(open('$O/split_fresh4.json'))['programs']['$cell'] or '')")
  echo "[$(date +%H:%M:%S)] start $cell program=${prog:-roots}"
  t0=$(date +%s)
  SUITE=$suite LEVEL=$level TASK=$task OUT=$out GPUS=$GPUS ROLLOUT_GPUS=$ROLLOUT_GPUS ENVS=40 SEED=0 \
    MAX_BRANCHES=3 INIT_ROOT=$INIT_ROOT PROGRAM=$prog timeout 7200 bash $TRIAL
  echo "[$(date +%H:%M:%S)] end $cell rc=$? $(( $(date +%s) - t0 ))s eps=$(find $out -name 'process-*.json' | wc -l)"
done
echo ALL_DONE
