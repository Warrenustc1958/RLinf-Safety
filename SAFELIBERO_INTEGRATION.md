# LIBERO-Safety integration

This integration runs OpenVLA-OFT evaluation and RL rollouts in the official
`LIBERO-Safety` simulator. The source checkout and Python environment live next
to `RLinf`, so simulator assets and dependencies do not enter the RLinf tree.

The supported simulator suites are `affordance`, `human_safety`,
`obstacle_avoidance`, and `obstacle_avoidance_human`. Each suite contains five
tasks at each of `L0`, `L1`, and `L2`. The repository also registers
`reasoning_safety`, but it has no simulator init-state directory and is not a
robot rollout suite.

## Install

Assume this layout:

```text
/path/to/workspace/
├── RLinf/
├── LIBERO-Safety/
├── checkpoints/
└── .venv-openvlaoft-libero-safety/
```

Install RLinf, OpenVLA-OFT, and the official simulator into a dedicated venv:

```bash
cd /path/to/workspace/RLinf
export LIBERO_SAFETY_PATH=/path/to/workspace/LIBERO-Safety

bash requirements/install.sh embodied \
  --model openvla-oft \
  --env liberosafety \
  --venv ../.venv-openvlaoft-libero-safety \
  --use-mirror \
  --no-root \
  --no-flash-attn
```

``--no-root`` is appropriate for a managed image that already contains the
EGL/MuJoCo system libraries; it also avoids changing a shared machine's apt
state. The supplied LIBERO-Safety configs use PyTorch SDPA, so
``--no-flash-attn`` does not require another override. The installer
deliberately does not install
`LIBERO-Safety/requirements.txt`: its old exact pins would downgrade the
RLinf/OpenVLA-OFT stack. It installs the official source tree and its bundled
robosuite fork without dependencies, then adds only the missing runtime
packages.

The official simulator assets are a separate 10.7 GB download. Download the
archive beside the repositories, then extract it exactly where LIBERO-Safety
expects `assets/`:

```bash
source /path/to/workspace/.venv-openvlaoft-libero-safety/bin/activate
hf download LIBERO-Safety/libero_safety_assets assets.zip \
  --repo-type dataset \
  --local-dir /path/to/workspace/LIBERO-Safety-assets

unzip /path/to/workspace/LIBERO-Safety-assets/assets.zip \
  -d /path/to/workspace/LIBERO-Safety/libero/libero/
test -d /path/to/workspace/LIBERO-Safety/libero/libero/assets
```

The installer puts LIBERO's config under the venv at
`.venv-openvlaoft-libero-safety/share/libero-safety/config.yaml`. This avoids
cross-contamination from the global `~/.libero/config.yaml` used by other
LIBERO forks.

Verify the package, benchmark, and paths before allocating GPUs:

```bash
export LIBERO_CONFIG_PATH=/path/to/workspace/.venv-openvlaoft-libero-safety/share/libero-safety
python - <<'PY'
import libero.libero as libero
from libero.libero.benchmark import get_benchmark

suite = get_benchmark("obstacle_avoidance")()
task = suite.get_task_by_level_id(0, 0)
print("benchmark root:", libero.get_libero_path("benchmark_root"))
print("tasks:", suite.get_task_distribution_by_level())
print("BDDL:", suite.get_task_bddl_file_path(0, 0))
print("init states:", len(suite.get_task_init_states(0, 0)))
print("first task:", task.language)
PY
```

## Evaluate OpenVLA-OFT

The launcher defaults Ray and process temporary files to
`/root/.cache/rlinf/libero-safety` instead of the container's small `/tmp`
filesystem. Set `RLINF_RUNTIME_DIR` to another filesystem with sufficient free
space when required.

`run_safelibero.sh` accepts all paths through environment variables. The
default model is the local RLinf LIBERO-130 base-plus-LoRA checkpoint; replace
it with a LIBERO-Safety-finetuned checkpoint when available.

```bash
cd /path/to/workspace/RLinf
MODE=eval \
SUITE=obstacle_avoidance \
LEVEL=L1 \
GPU_RANKS=0-3 \
TOTAL_ENVS=250 \
MODEL=/path/to/checkpoint \
IS_LORA=True \
LORA_PATH=/path/to/checkpoint/lora_adapter \
UNNORM_KEY=libero_130_no_noops_trajall \
bash run_safelibero.sh
```

There are normally 250 held-out episodes per suite and level (five tasks times
50 init states), so `TOTAL_ENVS=250` evaluates the complete split in one wave.
For a smoke test, use `TOTAL_ENVS=4` and optionally select one global task id:

```bash
MODE=eval SUITE=affordance LEVEL=L0 TOTAL_ENVS=4 GPU_RANKS=0 \
  bash run_safelibero.sh env.eval.task_id_filter='[0]'
```

## Run RL rollouts / GRPO

The rollout mode uses the same simulator adapter and OpenVLA-OFT action path,
and adds the actor worker for GRPO updates:

```bash
cd /path/to/workspace/RLinf
MODE=rollout \
SUITE=obstacle_avoidance_human \
LEVEL=L2 \
GPU_RANKS=0-3 \
MODEL=/path/to/checkpoint \
IS_LORA=True \
LORA_PATH=/path/to/checkpoint/lora_adapter \
UNNORM_KEY=libero_130_no_noops_trajall \
bash run_safelibero.sh
```

The environment reward comes from the task's BDDL goal predicates. For
OpenVLA-OFT, `UNNORM_KEY` must exist in the checkpoint's
`dataset_statistics.json`. Reusing LIBERO-130 statistics is suitable for a
zero-shot smoke test because the action space is the same, but training on a
new LIBERO-Safety demonstration set should compute and use its own action
statistics key.

## Adapter behavior

RLinf still exposes the environment as `env_type: libero`. Setting
`LIBERO_TYPE=safety` selects the installed official fork. The adapter then:

1. validates the four physical suites and `L0`–`L2`;
2. filters the benchmark's 15 tasks to the five tasks at the requested level;
3. resolves BDDL files through the official level-aware API;
4. loads init states through `get_task_init_states(level, level_id)`;
5. sends the standard 256×256 agent-view image, proprioceptive state, language
   instruction, and 7-D actions through RLinf's existing OpenVLA-OFT path.

The official BDDL predicates define success and safety semantics. The adapter
does not invent a generic collision metric, because collision meaning differs
between affordance, human-contact, obstacle, and semantic tasks.

## Fine-tune OpenVLA-OFT on the demonstrations

The demonstration release is a LeRobot-v2 tree. Its parquet files contain
state, 7-D delta action, timestamps, and task indices; RGB observations are not
embedded in parquet. The two image features in `meta/info.json` point to one
agent-view and one wrist-view MP4 per episode. The OpenVLA-OFT LIBERO recipe
uses the agent view only, so the wrist videos are optional unless
`data.video_keys` is changed to request them.

Start with a short loader/training smoke test:

```bash
cd /path/to/workspace/RLinf
DATASET=/path/to/workspace/datasets/libero_safety \
MODEL=/path/to/workspace/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora \
bash examples/sft/run_safelibero_openvlaoft.sh \
  runner.max_steps=2 runner.save_interval=2 \
  actor.global_batch_size=8 actor.micro_batch_size=1 \
  data.max_episodes=16
```

The loader forms an eight-step future action chunk, normalizes its first six
dimensions with `meta/stats.json` q01/q99 bounds, leaves the binary gripper
dimension unchanged, and applies action-token cross entropy. It continues the
LIBERO-130 LoRA adapter by default. Set `actor.model.lora_path=null` to create a
fresh adapter instead.

Partial video downloads are accepted when
`data.require_complete_videos=true`: startup logs the number of usable episodes
and indexes only those whose requested video files exist. This is useful for a
smoke test, but a full-data run needs every `observation.image` MP4. The wrist
half of the video release is not needed by the default one-camera checkpoint.
