#!/usr/bin/env bash
# fresh4 step 6: after the given PIDs (chain / earlier retry) exit, delete every cell dir without
# EXIT=0 (e.g. env workers died with a Gloo error, or the free-space guard stopped the run), take the
# LIBERO lock again and re-run those cells with a lower free-space guard.
#   MIN_FREE_GB=15 setsid nohup bash retry_fresh4.sh <pid>... > retry.log 2>&1 &
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
F=${FRESH4:-/data_cf/cws_sam/v1/fresh4}
ROOT=${FRESH4_CF:-/data_cf/cws/fresh4_cf}
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 120; done; done
echo "[$(date +%H:%M:%S)] waited for: $*"
for d in $ROOT/runs/*/; do
  grep -q "EXIT=0" "$d/launcher.log" 2>/dev/null && continue
  echo "retry $(basename $d) (had $(find "$d" -name 'process-*.json' | wc -l) episodes)"; rm -rf "$d"
done
exec 9>/tmp/rlinf-libero-run.lock
flock 9
echo "[$(date +%H:%M:%S)] lock acquired, retrying"
MIN_FREE_GB=${MIN_FREE_GB:-15} bash $HERE/collect_fresh4.sh > $F/collect_retry.log 2>&1
echo "[$(date +%H:%M:%S)] RETRY_DONE rc=$?"
