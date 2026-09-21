# SafeLIBERO integration

This branch adds SafeLIBERO evaluation support to RLinf's LIBERO environment.
SafeLIBERO itself is kept as a separate source tree and must be installed into
the same Python environment before running these configurations.

## Evaluation

The launcher accepts `goal`, `object`, `spatial`, `long`, or `libero_10`.
SafeLIBERO calls the four LIBERO-10-derived long-horizon tasks
`safelibero_long`; `long` and `libero_10` therefore select the same suite.

```bash
cd /path/to/RLinf
SUITE=goal SAFETY_LEVEL=I TOTAL_ENVS=40 GPU_RANKS=0-3 \
  bash run_safelibero.sh

# Full Object evaluation: four tasks x 50 held-out states
SUITE=object SAFETY_LEVEL=II TOTAL_ENVS=200 GPU_RANKS=0-3 \
  bash run_safelibero.sh
```

The local launcher defaults to a Goal-specific full SFT checkpoint for Goal.
For Object, Spatial, and Long it defaults to the LIBERO-130 base checkpoint
plus its LoRA adapter. Override `MODEL`, `UNNORM_KEY`, `IS_LORA`, and
`LORA_PATH` when using another checkpoint. The normalization key must exist in
the selected checkpoint's `dataset_statistics.json`.

Reported safety metrics include obstacle contact, obstacle displacement,
first collision step, collision step count, and collision-free task success
(`safe_success_once`).

## Demonstration collection

SafeLIBERO contains LIBERO's two-stage teleoperation tools:

```bash
cd /path/to/vlsa-aegis/safelibero
python scripts/collect_demonstration.py \
  --device keyboard \
  --num-demonstration 50 \
  --bddl-file libero/libero/bddl_files/safelibero_goal/put_the_bowl_on_the_plate.bddl \
  --directory demonstration_data

python scripts/create_dataset.py \
  --demo-file demonstration_data/<run>/demo.hdf5 \
  --use-camera-obs
```

`collect_demonstration.py` records raw simulator states and actions;
`create_dataset.py` deterministically replays them and writes LIBERO HDF5
observations (`agentview_rgb`, wrist RGB, proprioception, states, and actions).

For safety training, do not treat the stock collector's output as
collision-free automatically. The stock script accepts task success but does
not reject obstacle contact or displacement, and its random reset does not
select a SafeLIBERO Level-I/II held-out state. A production collector should:

1. generate a separate training set of initial states for each safety level;
2. reject demonstrations with obstacle contact or displacement;
3. retain rejected/colliding rollouts separately as negative reward-model data;
4. store suite, task, level, seed, obstacle pose, success, and collision labels;
5. reserve the repository's 50 `*.pruned_init` states per task for evaluation.

After collection, convert only the collision-free expert split to the RLDS
schema expected by OpenVLA-OFT and compute a new dataset-statistics key, for
example `safelibero_goal_level_i_no_noops`. Do not reuse a LIBERO normalization
key after action statistics have changed.
