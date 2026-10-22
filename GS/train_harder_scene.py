import math
import os
import torch
import imageio
import argparse
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import visualize_renders
from data_utils_harder_scene import get_nerf_datasets, trivial_collate

from pytorch3d.renderer import PerspectiveCameras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

MEANS_LR = 0.002

def ssim_loss(pred, gt, window_size=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    # Gaussian-window SSIM over the valid (unpadded) region
    x = pred.permute(2, 0, 1)[None]
    y = gt.permute(2, 0, 1)[None]
    c = x.shape[1]
    coords = torch.arange(window_size, dtype=x.dtype, device=x.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(c, 1, window_size, window_size).contiguous()
    blur = lambda t: F.conv2d(t, w, groups=c)
    mu_x, mu_y = blur(x), blur(y)
    s_xx = blur(x * x) - mu_x ** 2
    s_yy = blur(y * y) - mu_y ** 2
    s_xy = blur(x * y) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * s_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (s_xx + s_yy + C2))
    return ssim_map.mean()

def make_trainable(gaussians):

    gaussians.means.requires_grad_(True)
    gaussians.pre_act_scales.requires_grad_(True)
    gaussians.colours.requires_grad_(True)
    gaussians.pre_act_opacities.requires_grad_(True)

    if not gaussians.is_isotropic:
        gaussians.pre_act_quats.requires_grad_(True)

def setup_optimizer(gaussians, improved=False):
    gaussians.check_if_trainable()
    if improved:
        lrs = dict(opacities=0.05, scales=0.005, colours=0.025, means=MEANS_LR, quats=0.001)
    else:
        lrs = dict(opacities=0.05, scales=0.01, colours=0.05, means=0.001, quats=0.001)  # your Q6 set
    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': lrs["opacities"], "name": "opacities"},
        {'params': [gaussians.pre_act_scales], 'lr': lrs["scales"], "name": "scales"},
        {'params': [gaussians.colours], 'lr': lrs["colours"], "name": "colours"},
        {'params': [gaussians.means], 'lr': lrs["means"], "name": "means"},
    ]
    if not gaussians.is_isotropic:
        parameters.append({'params': [gaussians.pre_act_quats], 'lr': lrs["quats"], "name": "quats"})
    return torch.optim.Adam(parameters, lr=0.0, eps=1e-15)

def ndc_to_screen_camera(camera, img_size = (128, 128)):

    min_size = min(img_size[0], img_size[1])

    screen_focal = camera.focal_length * min_size / 2.0
    screen_principal = torch.tensor([[img_size[0]/2, img_size[1]/2]]).to(torch.float32)

    return PerspectiveCameras(
        R=camera.R, T=camera.T, in_ndc=False,
        focal_length=screen_focal, principal_point=screen_principal,
        image_size=(img_size,),
    )

def run_training(args):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name="materials", data_root=args.data_path,
        image_size=[128, 128],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    train_itr = iter(train_loader)

    # Preparing some code for visualization
    viz_gif_path_1 = os.path.join(args.out_path, "part_7_2_training_progress.gif")
    viz_gif_path_2 = os.path.join(args.out_path, "part_7_2_training_final_renders.gif")
    viz_idxs = np.linspace(0, len(train_dataset)-1, 5).astype(np.int32)[:4]

    gt_viz_imgs = [(train_dataset[i]["image"]*255.0).numpy().astype(np.uint8) for i in viz_idxs]
    gt_viz_imgs = [np.array(Image.fromarray(x).resize((256, 256))) for x in gt_viz_imgs]
    gt_viz_img = np.concatenate(gt_viz_imgs, axis=1)

    viz_cameras = [ndc_to_screen_camera(train_dataset[i]["camera"]).cuda() for i in viz_idxs]

    # Init gaussians and scene
    gaussians = Gaussians(
        num_points=10000, init_type="random",
        device=args.device, isotropic=not args.improved,
    )
    scene = Scene(gaussians)

    if args.improved:
        gaussians.means = (torch.rand_like(gaussians.means) * 2.0 - 1.0) * 1.2
        gaussians.pre_act_scales = torch.full_like(gaussians.pre_act_scales, math.log(0.02))
        gaussians.pre_act_opacities = torch.full_like(gaussians.pre_act_opacities, math.log(0.1 / 0.9))

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    optimizer = setup_optimizer(gaussians, improved=args.improved)

    # Training loop
    viz_frames = []
    for itr in range(args.num_itrs):
        if args.improved:
            for g in optimizer.param_groups:
                if g["name"] == "means":
                    g["lr"] = MEANS_LR * 0.01 ** (itr / args.num_itrs)
        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)

        gt_img = data[0]["image"].cuda()
        camera = ndc_to_screen_camera(data[0]["camera"]).cuda()

        # Rendering scene using gaussian splatting
        # HINT: Can any function from the Scene class help?
        # HINT: Set bg_colour to (0.0, 0.0, 0.0)
        # HINT: Set img_size to (128, 128)
        # HINT: Get per_splat from args.gaussians_per_splat
        # HINT: camera is available above
        pred_img, _, _ = scene.render(
            camera, 
            per_splat=args.gaussians_per_splat,
            img_size=(128, 128),
            bg_colour=(0.0, 0.0, 0.0),
        )

        # Compute loss
        l1 = F.l1_loss(pred_img, gt_img)
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim_loss(pred_img, gt_img)) if args.improved else l1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

        if itr % args.viz_freq == 0:
            viz_frame = visualize_renders(
                scene, gt_viz_img,
                viz_cameras, (128, 128)
            )
            viz_frames.append(viz_frame)

    print("[*] Training Completed.")

    # Saving training progess GIF
    imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1/10.0)*1000)

    # Creating renderings of the training views after training is completed.
    frames = []
    viz_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
        gt_img = viz_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(viz_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, _, _ = scene.render(
                camera, 
                per_splat=args.gaussians_per_splat,
                img_size=(128, 128),
                bg_colour=(0.0, 0.0, 0.0),
            )

        pred_npy = pred_img.detach().cpu().numpy()
        pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(pred_npy)

    # Saving renderings
    imageio.mimwrite(viz_gif_path_2, frames, loop=0, duration=(1/10.0)*1000)

    # Running evaluation using the test dataset
    psnr_vals, ssim_vals = [], []
    for val_data in tqdm(val_loader, desc="Running Evaluation"):

        gt_img = val_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(val_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, _, _ = scene.render(
                camera, 
                per_splat=args.gaussians_per_splat,
                img_size=(128, 128),
                bg_colour=(0.0, 0.0, 0.0),
            )

            gt_npy = gt_img.detach().cpu().numpy()
            pred_npy = pred_img.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
            ssim = structural_similarity(gt_npy, pred_npy, channel_axis=-1, data_range=1.0)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
    print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/materials", type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )
    parser.add_argument(
        "--num_itrs", default=1000, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=20, type=int,
        help="Frequency with which visualization should be performed."
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    parser.add_argument("--improved", action="store_true", help="for harder scenes; off = Q6 baseline")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)
