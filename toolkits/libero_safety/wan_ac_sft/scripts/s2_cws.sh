#!/usr/bin/env bash
# Stage 1+2 on the CWS index: fp32 master weights, real batches, EMA, grad clip,
# warmup + cosine. GPUs 2,3,6,7. Usage: s2_cws.sh <output_dir> [extra trainer args...]
# Relaunches after a failure (resume from <output_dir>/progress.json), max 20 tries.
set -uo pipefail
OUTDIR=$1; shift
W=/mnt/nfs/fjj/wan_ac_sft
C=$W/ckpt/RLinf-Wan-LIBERO-Spatial
CFG=$W/scripts/accelerate_4gpu_cws.yaml
cat > $CFG <<YAML
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 4
machine_rank: 0
main_process_port: 29531
mixed_precision: bf16
main_training_function: main
rdzv_backend: static
same_network: true
use_cpu: false
YAML
cd /mnt/nfs/fjj/Code/RLinf-Safety
export PYTHONPATH=/mnt/nfs/fjj/Code/related_works/diffsynth-studio:.
export CUDA_VISIBLE_DEVICES=2,3,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
for attempt in $(seq 1 20); do
  [ -f "$OUTDIR/DONE" ] && break
  echo "[sup] attempt $attempt $(date "+%F %T")"
  $W/venv-wan/bin/python -m accelerate.commands.launch --config_file $CFG \
    -m toolkits.libero_safety.wan_ac_sft.train_wan_cws \
    --index /data_cf/wan_ac_sft/index/cws_v1 --init_dit $C/model-00001.safetensors --vae $C/Wan2.2_VAE.pth \
    --output "$OUTDIR" "$@"
  rc=$?; echo "[sup] attempt $attempt exit $rc $(date "+%F %T")"
  [ -f "$OUTDIR/DONE" ] && break
  [ "${SMOKE:-0}" = 1 ] && break
  sleep 60
done
echo "[sup] DONE $(date "+%F %T")"
