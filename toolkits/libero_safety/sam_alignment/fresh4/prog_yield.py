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

"""fresh4 step 1: per-cell unsafe yield of archived Step E branch programs.

Counts admitted branches by quadrant in the Step E archive outcomes and writes the
program with the highest Q2+Q4 rate per cell to <fresh4>/best_programs.json.
"""

import json
from collections import defaultdict
from pathlib import Path

ARCHIVE = Path("/data_cf/cws/step_e/archive")
F = Path("/data_cf/cws_sam/v1/fresh4")


def main():
    stat = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for line in open(ARCHIVE / "outcomes.jsonl"):
        o = json.loads(line)
        if o.get("kind") != "branch":
            continue
        s = stat[o["cell"]][o["program_id"]]
        s["n"] += 1
        if o.get("admitted"):
            s["adm"] += 1
            s[o["quadrant"]] += 1
    best = {}
    for cell in sorted(stat):
        rows = []
        for pid, s in stat[cell].items():
            unsafe = s["Q2"] + s["Q4"]
            rows.append((unsafe / max(s["n"], 1), unsafe, s["n"], s["Q1"] + s["Q3"], pid))
        rows.sort(reverse=True)
        exists = [r for r in rows if (ARCHIVE / "programs" / f"{r[4]}.json").exists()]
        top = exists[0] if exists else None
        best[cell] = None if top is None else {"program": top[4], "unsafe_rate": round(top[0], 3),
                                               "unsafe": top[1], "branches": top[2], "safe": top[3]}
        print(cell, len(rows), best[cell])
    F.mkdir(parents=True, exist_ok=True)
    json.dump(best, open(F / "best_programs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
