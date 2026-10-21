"""GPU-5 evaluation chain for the stage 1+2 CWS run.

1. Baselines with metrics v2 (standard FID / FVD, FloLPIPS) and, where missing,
   16-chunk controllability (ar_eval) on the v1 test set.
2. For every EMA export of runs/s2_cws_b20 at a multiple of --every steps: hard-link
   it (so checkpoint rotation cannot delete it mid-eval), run ar_eval + metrics v2,
   then drop the link unless the trainer keeps that milestone itself.
Results go to /mnt/nfs/fjj/wan_ac_sft/runs/eval_v2/<tag>_{ar,mv2}.json.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

W = "/mnt/nfs/fjj/wan_ac_sft"
REPO = "/mnt/nfs/fjj/Code/RLinf-Safety"
VAE = f"{W}/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth"
INDEX = f"{W}/index/v1"
OUT = Path(f"{W}/runs/eval_v2")
RUN = Path("/data_cf/wan_ac_sft/runs/s2_cws_b20")
BASELINES = [  # tag, dit, pose_scale, pose_clip, need_ar
    ("rlinf_zeroshot", f"{W}/ckpt/RLinf-Wan-LIBERO-Spatial/model-00001.safetensors", 1.0, 0.0, False),
    ("s1_cont_60k", f"{W}/runs/s1_opt3_cont/step-60120.safetensors", 1.0, 0.0, False),
    ("mp_30060", "/data_cf/wan_ac_sft/runs/s1_motion_pose/step-30060.safetensors", 8.0, 4.0, False),
    ("epoch1_190k", f"{W}/runs/s1_epoch1/step-190516.safetensors", 1.0, 0.0, True),
]


def run(cmd, log):
    env = dict(os.environ, PYTHONPATH="/mnt/nfs/fjj/Code/related_works/diffsynth-studio:.",
               CUDA_VISIBLE_DEVICES=os.environ.get("EVAL_GPU", "5"))
    with open(log, "a") as f:
        f.write(f"\n[chain] {time.strftime('%F %T')} {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"[chain] exit {rc} {time.strftime('%F %T')}\n")
    return rc


def evaluate(tag, dit, ps, pc, need_ar, log):
    py = f"{W}/venv-wan/bin/python"
    if need_ar and not (OUT / f"{tag}_ar.json").exists():
        run([py, "-m", "toolkits.libero_safety.wan_ac_sft.ar_eval", "--index", INDEX, "--dit", dit, "--vae", VAE,
             "--out", str(OUT / f"{tag}_ar.json"), "--per-quadrant", "5", "--chunks", "16", "--seeds", "3",
             "--steps", "5", "--pose_scale", str(ps), "--pose_clip", str(pc)], log)
    if not (OUT / f"{tag}_mv2.json").exists():
        run([py, "-m", "toolkits.libero_safety.wan_ac_sft.wovr_metrics_v2", "--index", INDEX, "--dit", dit,
             "--vae", VAE, "--out", str(OUT / f"{tag}_mv2.json"), "--horizons", "128,256", "--per-quadrant", "16",
             "--steps", "5", "--pose_scale", str(ps), "--pose_clip", str(pc), "--tag", tag], log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=40000)
    ap.add_argument("--keep_every", type=int, default=10000, help="trainer keeps these EMA files itself")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = str(OUT / "chain.log")
    for b in BASELINES:
        if Path(b[1]).exists():
            evaluate(*b, log=log)
    hold = OUT / "hold"
    hold.mkdir(exist_ok=True)
    for k in range(a.every, a.max_steps + 1, a.every):
        src = RUN / f"ema-step-{k}.safetensors"
        while not src.exists():
            time.sleep(60)
        time.sleep(30)  # the export is written atomically; wait for the resume save to settle
        link = Path("/data_cf/wan_ac_sft/runs") / f".eval_hold_ema-{k}.safetensors"
        if not link.exists():
            os.link(src, link)
        evaluate(f"cws_ema_{k}", str(link), 8.0, 4.0, True, log)
        link.unlink()
    with open(log, "a") as f:
        f.write(f"[chain] ALL DONE {time.strftime('%F %T')}\n")


if __name__ == "__main__":
    main()
