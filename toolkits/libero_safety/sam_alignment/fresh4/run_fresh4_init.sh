#!/usr/bin/env bash
# fresh4 step 3: 40 fresh init states per task, seed base 30261025 (no overlap with fresh_init/v1).
# Two suites per GPU (render only), written to <out>_part_a / <out>_part_b; chain_fresh4.sh merges them.
#   GPU_A=6 GPU_B=7 bash run_fresh4_init.sh
# gpu006: GPUs 3 and 4 fault in EGL; part a once exited silently on GPU 5.
set -uo pipefail
S=/mnt/nfs/fjj/Code/CodeWorld-Safe/toolkits/codeworld_safe/cf/run_fresh_init.sh
O=${OUT:-/data_cf/cws/fresh_init/v2}
mkdir -p "$(dirname $O)/logs4"
GPU=${GPU_A:-6} bash $S --out ${O}_part_a --n 40 --seed_base 30261025 --suites affordance,human_safety \
  > "$(dirname $O)/logs4/a.log" 2>&1 &
GPU=${GPU_B:-7} bash $S --out ${O}_part_b --n 40 --seed_base 30261025 --suites obstacle_avoidance,obstacle_avoidance_human \
  > "$(dirname $O)/logs4/b.log" 2>&1 &
wait
echo PARTS_DONE
