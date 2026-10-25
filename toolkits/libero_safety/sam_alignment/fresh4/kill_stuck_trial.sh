#!/usr/bin/env bash
# End one stuck fresh4 trial (the `timeout ... trial_step_e.sh` child of collect_fresh4.sh) by PID
# tree, so collect_fresh4.sh moves on to the next cell. No `ray stop`: other jobs are untouched.
# The killed cell has no EXIT=0 and is picked up by retry_fresh4.sh.
desc() { for c in $(ps -o pid= --ppid "$1"); do echo "$c"; desc "$c"; done; }
C=$(ps -eo pid,args | grep "[c]ollect_fresh4" | awk '{print $1}' | head -1)
T=$(ps -o pid=,args= --ppid "$C" | grep "timeout" | awk '{print $1}' | head -1)
[ -n "$T" ] || { echo "no trial under collect pid $C"; exit 1; }
ALL="$T $(desc "$T")"
echo "collect $C trial $T, processes $(echo $ALL | wc -w)"
kill -TERM $ALL 2>/dev/null
sleep 15
for p in $ALL; do kill -0 "$p" 2>/dev/null && kill -KILL "$p"; done
