# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""fresh4 step 2: widened branch programs + group split for the one-shot collection.

Programs: the best archived Step E program per cell (best_programs.json), every
uniform param low bound scaled by 0.25, so sibling branches of one root range from
a small nudge to the original magnitude (mixed safe/unsafe outcomes -> hard
same-state pairs).
Split by fresh trial_id: 0-9 cost_hard, 10-19 h_deep, 20-29 calib_fresh,
30-39 test_fresh; branches follow their root. Writes <fresh4>/split_fresh4.json.
"""

import json
from pathlib import Path

ARCHIVE = Path("/data_cf/cws/step_e/archive/programs")
F = Path("/data_cf/cws_sam/v1/fresh4")
SUITES = ("affordance", "human_safety", "obstacle_avoidance", "obstacle_avoidance_human")
ROLES = ("cost_hard", "h_deep", "calib_fresh", "test_fresh")


def main():
    best = json.load(open(F / "best_programs.json"))
    (F / "programs").mkdir(exist_ok=True)
    cells = [f"{s}_L{lv}_t{t}" for s in SUITES for lv, ts in ((0, range(5)), (1, range(5, 10))) for t in ts]
    plan = {}
    for cell in cells:
        b = best.get(cell)
        if not b:
            plan[cell] = None
            continue
        p = json.load(open(ARCHIVE / f"{b['program']}.json"))
        p["program_id"] = p["program_id"] + "_wide"
        for v in p.get("params", {}).values():
            if isinstance(v, dict) and v.get("dist") == "uniform":
                is_int = isinstance(v["low"], int) and isinstance(v["high"], int)
                v["low"] = max(1, int(round(v["low"] * 0.25))) if is_int else round(v["low"] * 0.25, 4)
        p["rationale"] = "fresh4 widened copy (low bounds x0.25) of " + b["program"]
        dst = F / "programs" / f"{p['program_id']}.json"
        dst.write_text(json.dumps(p, indent=1))
        plan[cell] = str(dst)
    json.dump({"init_states_root": "/data_cf/cws/fresh_init/v2", "seed_base": 30261025, "n_per_task": 40,
               "rule": "trial_id 0-9 cost_hard, 10-19 h_deep, 20-29 calib_fresh, 30-39 test_fresh; branches follow root",
               "group_key": "fresh4:<suite>:<level>:<task_id>:<trial_id>",
               "trial_role": {t: ROLES[t // 10] for t in range(40)}, "programs": plan},
              open(F / "split_fresh4.json", "w"), indent=1)
    print(sum(v is not None for v in plan.values()), "cells with programs of", len(cells))


if __name__ == "__main__":
    main()
