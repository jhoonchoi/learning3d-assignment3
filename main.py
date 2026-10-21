"""Entry point for the assignment code submission.

Running this file should reproduce every result referenced in the webpage
writeup (proj3/web/assignment.md). Each question runs the starter script from its
own folder (nerf/ uses hydra configs, GS/ uses argparse), as the handout does.

    python main.py                      # everything, in order
    python main.py render_box train_box # selected questions only

Generated using claude
"""
import argparse
import os
import shutil
import subprocess
import sys

import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
NERF_DIR = os.path.join(ROOT, "nerf")
GS_DIR = os.path.join(ROOT, "GS")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


"""
1.3-1.5. Differentiable volume rendering (30 points)
"""
def render_box():
    # images/grid.png, rays.png, sample_points.png (Q1.3-1.4), part_1.gif + depth (Q1.5)
    run_nerf("box")


"""
2.1-2.3. Optimizing a basic implicit volume (10 points)
"""
def train_box(num_epochs=1000):
    # prints the optimized box center and side lengths; images/part_2.gif
    run_nerf("train_box", [f"training.num_epochs={num_epochs}"])


"""
3. Optimizing a Neural Radiance Field (20 points)
"""
def train_nerf_lego(num_epochs=250, retrain=False):
    train_nerf("nerf_lego", tag="lego", gif="part_3.gif",
               num_epochs=num_epochs, retrain=retrain)


"""
4.1. View dependence (10 points)
"""
def train_nerf_materials(num_epochs=250, highres=False, retrain=False):
    # the handout asks for materials results on the webpage (lego is only for testing)
    tag = "materials_highres" if highres else "materials"
    train_nerf(f"nerf_{tag}", tag=tag, gif=f"part_4_1_{tag}.gif",
               num_epochs=num_epochs, retrain=retrain)


"""
5.1-5.5. 3D Gaussian rasterization (35 points)
"""
def test_gaussians():
    # 4 unit tests for the projection and 2D evaluation (Q5.1-5.2), CPU only
    run_gs("unit_test_gaussians.py")


def render_gaussians(gaussians_per_splat=2048):
    # output/part_5/part_5_render.gif of the pre-trained chair
    run_gs("render.py", ["--out_path", "output/part_5",
                         "--gaussians_per_splat", str(gaussians_per_splat)])


"""
6. Training 3D Gaussian representations (15 points)
"""
def train_gaussians(num_itrs=1000, gaussians_per_splat=-1):
    # output/part_6/ training progress + final render GIFs; prints PSNR and SSIM
    run_gs("train.py", ["--out_path", "output/part_6", "--num_itrs", str(num_itrs),
                        "--gaussians_per_splat", str(gaussians_per_splat)])


"""
7.1. Rendering using spherical harmonics (10 points)
"""
def render_gaussians_sh(gaussians_per_splat=2048):
    # TODO (Q7.1): render.py has no switch between DC-only (Q5.5) and full SH yet;
    # add one when implementing [Q 7.1] so both GIFs can be reproduced from here
    run_gs("render.py", ["--out_path", "output/part_7_1",
                         "--gaussians_per_splat", str(gaussians_per_splat)])


"""
Helper functions
"""
def train_nerf(config, tag, gif, num_epochs, retrain=False):
    # every config defaults to the same ./checkpoints file with resume: True, so each
    # run gets its own checkpoint; otherwise materials would resume from lego's weights
    checkpoint = os.path.join("checkpoints", f"{tag}.pth")
    output = os.path.join(NERF_DIR, "images", gif)

    # skip if already trained; a finished checkpoint resumes past the last render
    if retrain and os.path.exists(os.path.join(NERF_DIR, checkpoint)):
        os.remove(os.path.join(NERF_DIR, checkpoint))
    if os.path.exists(os.path.join(NERF_DIR, checkpoint)) and os.path.exists(output):
        print(f"[{tag}] {checkpoint} and {gif} exist, skipping (retrain=True to redo)")
        return

    run_nerf(config, [f"training.num_epochs={num_epochs}",
                      f"training.checkpoint_path={checkpoint}"])

    # train_nerf always writes images/part_3.gif; keep a per-run copy
    if gif != "part_3.gif":
        shutil.copy(os.path.join(NERF_DIR, "images", "part_3.gif"), output)


def run_nerf(config, overrides=()):
    os.makedirs(os.path.join(NERF_DIR, "images"), exist_ok=True)
    run([sys.executable, "volume_rendering_main.py", f"--config-name={config}", *overrides],
        cwd=NERF_DIR)


def run_gs(script, args=()):
    if script != "unit_test_gaussians.py":
        args = [*args, "--device", DEVICE]
    run([sys.executable, script, *args], cwd=GS_DIR)


def run(cmd, cwd):
    print(f"\n$ (cd {os.path.relpath(cwd, ROOT)} && {' '.join(cmd[1:])})", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


PARTS = {
    "render_box": render_box,
    "train_box": train_box,
    "train_nerf_lego": train_nerf_lego,
    "train_nerf_materials": train_nerf_materials,
    "test_gaussians": test_gaussians,
    "render_gaussians": render_gaussians,
    "train_gaussians": train_gaussians,
    "render_gaussians_sh": render_gaussians_sh,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="*", metavar="part",
                        help=f"questions to run (default: all, in order): {', '.join(PARTS)}")
    args = parser.parse_args()
    unknown = [p for p in args.parts if p not in PARTS]
    if unknown:
        parser.error(f"unknown part(s) {unknown}; choose from {list(PARTS)}")

    # A. Neural volume rendering: 1.3-1.5, 2, 3, 4.1 (view dependence)
    # B. 3D Gaussian splatting: 5.1-5.5, 6, 7.1 (spherical harmonics)
    for name in args.parts or PARTS:
        PARTS[name]()
    return


if __name__ == "__main__":
    main()
