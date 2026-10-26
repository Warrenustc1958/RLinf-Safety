#!/usr/bin/env python3
"""fresh4 step 9: upload the fresh4 collection to the CWS ModelScope dataset repo and verify every file.

Repo: Warrenustc1958/CodeWorld-Safe-pi05-counterfactual-rollout (dataset), new top-level folder fresh4/:
  fresh4/runs/<cell>/        raw audits + visual npz (/data_cf/cws/fresh4_cf/runs), one unit per cell
  fresh4/init_states_v2/     the 40 x 40 fresh init states (/data_cf/cws/fresh_init/v2)
  fresh4/meta/               split_fresh4.json, best_programs.json, widened programs, manifest, grader labels
Files are staged as per-file symlinks (nothing copied). Never deletes remote or source files.
Units that verify (same path set and sizes as local) are skipped on rerun.

  MS_TOKEN_FILE=<600 file> /usr/bin/python3 upload_fresh4.py
"""
import json
import os
import sys
import time
from pathlib import Path

from modelscope.hub.api import HubApi

REPO = "Warrenustc1958/CodeWorld-Safe-pi05-counterfactual-rollout"
STAGE = Path(os.environ.get("FRESH4_STAGE", "/dev/shm/cws_ms_stage_fresh4"))
RUNS = Path("/data_cf/cws/fresh4_cf/runs")
INIT = Path("/data_cf/cws/fresh_init/v2")
F = Path("/data_cf/cws_sam/v1/fresh4")
META = ("split_fresh4.json", "best_programs.json", "programs", "index", "labels")
SKIP = {".ms_upload_cache", ".ms_upload_progress", "__pycache__"}
LOG = STAGE / "upload.log"
MAX_TRIES = 5

README = """# fresh4/ - one-shot fresh-init-state collection (SAM calibration / test)

Code: RLinf-Safety toolkits/libero_safety/sam_alignment/fresh4/. Same collector as step_e
(pi0.5, chunk 8, <= 512 steps, agentview + wrist, up to 3 counterfactual branches per root).

- init_states_v2/: 40 new init states per task (40 tasks), seed_base 30261025, post-settle snapshots,
  never used by Wan / RM / SAM training (distinct from every shipped state).
- runs/<cell>/: audits (json) + visual npz. One widened program per cell (best step_e program, uniform
  param low bounds x0.25), so sibling branches of a root range from a small nudge to the original size.
  affordance_L0_t2 / obstacle_avoidance_L0_t2: roots only (no program).
- meta/split_fresh4.json: role by fresh trial_id: 0-9 cost_hard, 10-19 h_deep, 20-29 calib_fresh,
  30-39 test_fresh; branches follow their root. Group key trial = 1000 + fresh trial_id.
- meta/index/fresh4/manifest.jsonl: cws-style manifest (4721 episodes: 1600 roots, 3121 branches).
- meta/labels/: per-step 3-channel grader labels (build_labels), episodes.jsonl, report.json.
calib_fresh is for threshold calibration only; test_fresh is evaluated once after the design is frozen.
"""


def say(msg):
    line = f"[ms {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def link(dst: Path, src: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    if dst.is_symlink() and os.readlink(dst) == str(src):
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def mirror(src: Path, dst: Path):
    if src.is_file():
        link(dst, src)
        return
    for p in src.rglob("*"):
        if p.is_file() and not (SKIP & set(p.parts)):
            link(dst / p.relative_to(src), p)


def stage():
    units = []
    for cell in sorted(RUNS.iterdir()):
        if cell.is_dir():
            mirror(cell, STAGE / "fresh4" / "runs" / cell.name)
            units.append(f"fresh4/runs/{cell.name}")
    mirror(INIT, STAGE / "fresh4" / "init_states_v2")
    units.append("fresh4/init_states_v2")
    for name in META:
        mirror(F / name, STAGE / "fresh4" / "meta" / name)
    readme = STAGE / "fresh4" / "meta" / "README.md"
    if readme.is_symlink():
        readme.unlink()
    readme.write_text(README)
    units.append("fresh4/meta")
    return units


def remote_files(api, root):
    out, page = {}, 1
    while True:
        try:
            fs = api.get_dataset_files(REPO, root_path=root, recursive=True, page_number=page, page_size=500)
        except Exception:  # folder not created yet
            return out
        for f in fs:
            if f.get("Type", "blob") != "tree":
                out[f["Path"]] = int(f.get("Size", -1))
        if len(fs) < 500:
            return out
        page += 1


def upload_unit(api, tok, prefix, state):
    sdir = STAGE / prefix
    local = {f"{prefix}/{p.relative_to(sdir).as_posix()}": p.stat().st_size
             for p in sdir.rglob("*") if p.is_file() and not (SKIP & set(p.parts))}
    remote = remote_files(api, prefix)
    todo = [k for k, s in local.items() if remote.get(k) != s]
    if todo:
        say(f"upload {prefix}: {len(todo)} of {len(local)} files, {sum(local[k] for k in todo) / 1e9:.2f} GB")
        api.upload_folder(REPO, folder_path=str(sdir), path_in_repo=prefix, repo_type="dataset",
                          commit_message=f"Upload {prefix}", token=tok)
        remote = remote_files(api, prefix)
    bad = [k for k, s in local.items() if remote.get(k) != s]
    ok = len(local) > 0 and not bad
    state[prefix] = {"files": len(local), "bytes": sum(local.values()), "bad": len(bad), "ok": ok,
                     "bad_examples": bad[:5], "checked": time.strftime("%F %T")}
    say(f"{'verified' if ok else 'MISMATCH'} {prefix}: {len(local)} files, "
        f"{sum(local.values()) / 1e9:.2f} GB, {len(bad)} bad")
    return ok


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    tok = Path(os.environ["MS_TOKEN_FILE"]).read_text().strip()
    api = HubApi()
    api.login(tok)
    state_f = STAGE / "state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    units = stage()
    say(f"staged {len(units)} units")
    for attempt in range(1, MAX_TRIES + 1):
        for prefix in units:
            if state.get(prefix, {}).get("ok"):
                continue
            try:
                upload_unit(api, tok, prefix, state)
            except Exception as e:  # network / rate limit: retry next pass
                state[prefix] = {"ok": False, "error": repr(e)[:300], "checked": time.strftime("%F %T")}
                say(f"FAILED {prefix}: {e!r}"[:400])
            state_f.write_text(json.dumps(state, indent=1))
        n_ok = sum(1 for p in units if state.get(p, {}).get("ok"))
        gb = sum(state[p].get("bytes", 0) for p in units if state.get(p, {}).get("ok")) / 1e9
        say(f"pass {attempt} done: {n_ok}/{len(units)} units verified, {gb:.1f} GB")
        if n_ok == len(units):
            say("ALL VERIFIED")
            return 0
        time.sleep(300)
    say("gave up after retries; rerun to continue")
    return 1


if __name__ == "__main__":
    sys.exit(main())
