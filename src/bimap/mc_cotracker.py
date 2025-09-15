"""Motion correction using CoTracker."""

import gc
import math
import os
import time

import numpy as np
import torch
from cotracker.utils.visualizer import Visualizer
from scipy.ndimage import map_coordinates, zoom
from utils import evaluate, find_highest_correlation, load_video, save_and_display_video

os.environ["TORCH_HOME"] = "/data/ih26ykel/cache/cotracker"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> tuple[np.ndarray, dict, float]:
    """Use this function only for local testing purposes, you probably want to use run() instead."""
    path = "../../data/input/strong_movement/b5czi.tif"
    video, frames, filename = load_video(path, len=400)
    return _run(video, frames, filename, "../../data/output/cotracker")


def run(config:dict) -> dict:
    """Run CoTracker motion correction on a video input.

    :param config: configuration dictionary with the following fields:
        data: path: path to the input video
        run:
            artifacts_dir: directory to save output artifacts
        diff_warp: (optional) whether to enforce diffeomorphic warping, default is False
        visibility: (optional) whether to use visibility weights from CoTracker, default is False
        grid_size: (optional) grid size for CoTracker, default is 15
        template_strategy: (optional) strategy to select the template frame,
                            either "first" or "computed", default is "first"
        gaussian_filtered: (optional) whether to apply Gaussian filtering
    :return: dictionary with results and metrics
    """
    path = config["data"]["path"]
    output_path = config["run"]["artifacts_dir"]
    filtered = config.get("gaussian_filtered", False)
    video, frames, filename = load_video(path, len = 400, gaussian_filtered=filtered)
    warped, metrics, runtime = _run(video, frames, filename, config)

    #bookkeping
    ssim_list = metrics["ssims"]
    mse_list = metrics["mse_list"]
    crispness_improvement = metrics["crispness_improvement"]
    metrics = {
        "per_frame": {
            "ssim": ssim_list,
            "mse": mse_list,
        },
        "summary": {
            "mse_mean": float(np.mean(mse_list)),
            "mse_std": float(np.std(mse_list)),
            "crispness_improvement": crispness_improvement,
        },
    }
    return {"runtime_s": runtime,
              "metrics": metrics,
              "artifacts": {
                  "output_path": f"{output_path}/{filename}.mp4",
              }}


def _run(video: np.ndarray, frames: list[np.ndarray], filename: str, config: dict) \
            -> tuple[np.ndarray, dict, float]:
    """Run internal CoTracker image registration.

    :param video: 4D numpy array representing the video (T, H, W, C)
    :param frames: list of 2D numpy arrays representing the video frames
    :param filename: name of the input video file without extension
    :param config: configuration dictionary containing at least the artifacts_dir

    :return: tuple of (motion_corrected_images, metrics, runtime)
        motion_corrected_images: list of motion corrected frames
        metrics: dictionary with evaluation metrics
        runtime: time taken to perform the registration
    """
    output_path = config["run"]["artifacts_dir"]
    visibility = config.get("visibility", False)
    diff_warp = config.get("diff_warp", False)
    grid_size = config.get("grid_size", 15)
    template_index = find_highest_correlation(frames) if config.get(
        "template_strategy") == "computed" else 0
    model = torch.hub.load("facebookresearch/co-tracker",
                           "cotracker3_offline").to(device)
    video = torch.from_numpy(video).to(device)
    model.model.model_resolution = video.shape[3:]
    start_time = time.time()
    pred_tracks, pred_visibility = model(video=video,
                                         grid_size=grid_size,
                                         grid_query_frame=template_index,
                                         backward_tracking=True)
    # save video with tracked points
    vis = Visualizer(save_dir=output_path)
    vis.visualize(video, pred_tracks, pred_visibility)

    result = warping(pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy(),
                     np.expand_dims(frames, axis=-1),
                     diff_warp=diff_warp, visibility=visibility)
    end_time = time.time()
    save_and_display_video(np.array(result), f"{output_path}/{filename}.mp4")
    torch.cuda.empty_cache()
    gc.collect()
    metrics = evaluate(result, frames, frames[template_index])
    return result, metrics, end_time - start_time


def warping(predicted_tracks: np.ndarray, predicted_visibility: np.ndarray,
            frames: np.ndarray, *, diff_warp: bool, visibility: bool) -> np.ndarray:
    """Warps the frames of a video based on predicted tracks.

    :param predicted_tracks: Predicted tracks of shape (B, T, G, D).
    :param predicted_visibility: Predicted visibility of shape (B, T, G).
    :param frames: Video frames of shape (T_orig, H, W, C).
    :param diff_warp: Whether to enforce diffeomorphic warping.
    :param visibility: Whether to use visibility weights from CoTracker.
    :return List of warped frames as numpy array

    """
    t_orig, h, w, c = frames.shape
    b, t, g, d = predicted_tracks.shape
    grid_size = int(math.sqrt(g))
    velocity = predicted_tracks[0].reshape(t, grid_size, grid_size,
                                           2)  # (24, 32, 32, 2)
    real_velocity = velocity - velocity[0]  # (24, 32, 32, 2)
    if visibility:
        visibility = predicted_visibility[0].reshape(t, grid_size, grid_size)
        vis_mask = (visibility > 0)
        # zero out points that cotracker deems unreliable
        real_velocity *= vis_mask[..., None]
    v = real_velocity.transpose(0, 3, 1, 2)  # (24, 2, 32, 32)
    vp = zoom(v, (1, 1, w / grid_size, h / grid_size))  # (24, 2, 256, 256)

    warpeds = [frames[0][..., 0]]

    for i in range(1, t_orig):
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        phi = np.diff(vp, axis=0)[0:i].sum(0)
        if diff_warp:
            phi = make_monotonic(phi)
        grid_x += phi[0].T
        grid_y += phi[1].T

        warped = map_coordinates(frames[i][..., 0].astype(np.float32), [grid_y, grid_x],
                                 order=3, mode="nearest")
        warpeds.append(warped)

    return np.array(warpeds)


def make_monotonic(phi: np.ndarray, eps: float=1e-3) -> np.ndarray:
    """Make a 2D displacement field monotonic by clamping negative increments.

        :param phi: (2, H, W) displacement field
        :param eps: small constant to avoid zero increments
    returns:
        diffeomorphic phi (smooth field)
    """
    # returns a new phi where each row/col displacement is non-decreasing
    dx = np.diff(phi[0], axis=1)       # shape (H, W-1)
    dy = np.diff(phi[1], axis=0)       # shape (H-1, W)
    # clamp negative steps
    dx_clamped = np.maximum(dx, eps)
    dy_clamped = np.maximum(dy, eps)
    # rebuild phi by cumulative sum of the (clamped) increments
    phi_x = np.concatenate((phi[0][:, :1], np.cumsum(dx_clamped, axis=1) + phi[0][:, :1]), axis=1)
    phi_y = np.concatenate((phi[1][:1, :], np.cumsum(dy_clamped, axis=0) + phi[1][:1, :]), axis=0)
    return np.stack((phi_x, phi_y), axis=0)


if __name__ == "__main__":
    main()
