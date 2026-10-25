#!/usr/bin/env bash
# fresh4 step 4: wait for both init parts (20 tasks each), merge into <INIT>/manifest.json, then take
# /tmp/rlinf-libero-run.lock (blocking) for the whole collection so nothing interleaves.
# WAIT_TMUX=<session> additionally waits until that tmux session has exited.
#   setsid nohup bash chain_fresh4.sh > chain.log 2>&1 &
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
O=${INIT_ROOT:-/data_cf/cws/fresh_init/v2}
F=${FRESH4:-/data_cf/cws_sam/v1/fresh4}
n_of() { python3 -c "import json; print(len(json.load(open('$1'))))" 2>/dev/null || echo 0; }
while true; do
  na=$(n_of ${O}_part_a/manifest.json); nb=$(n_of ${O}_part_b/manifest.json)
  echo "[$(date +%H:%M:%S)] init part_a $na/20 part_b $nb/20"
  [ "$na" -ge 20 ] && [ "$nb" -ge 20 ] && break
  sleep 120
done
if [ ! -f $O/manifest.json ]; then
  mkdir -p $O
  cp -r ${O}_part_a/affordance ${O}_part_a/human_safety ${O}_part_b/obstacle_avoidance \
    ${O}_part_b/obstacle_avoidance_human $O/
  python3 - "$O" <<'EOF' || { echo MERGE_FAILED; exit 1; }
import json, sys
O = sys.argv[1]
m = {}
for p in ("_part_a", "_part_b"):
    m.update(json.load(open(f"{O}{p}/manifest.json")))
assert len(m) == 40 and all(v["n"] == 40 for v in m.values()), len(m)
json.dump(m, open(f"{O}/manifest.json", "w"), indent=1)
print("merged", len(m))
EOF
fi
if [ -n "${WAIT_TMUX:-}" ]; then
  while tmux has-session -t "$WAIT_TMUX" 2>/dev/null; do
    echo "[$(date +%H:%M:%S)] waiting: tmux $WAIT_TMUX still running"; sleep 300
  done
fi
mkdir -p /data_cf/cws/fresh4_cf
exec 9>/tmp/rlinf-libero-run.lock
flock 9
echo "[$(date +%H:%M:%S)] lock acquired, collecting"
bash $HERE/collect_fresh4.sh > $F/collect.log 2>&1
echo "[$(date +%H:%M:%S)] CHAIN_DONE rc=$?"
