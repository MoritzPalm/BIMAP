import math
import time

import numpy as np
import torch
from scipy.ndimage import zoom, map_coordinates
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.models.core.model_utils import get_points_on_a_grid

from utils import load_video, save_and_display_video, evaluate, find_highest_correlation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    path = "../../data/input/strong_movement/b5czi.tif"
    video, frames, filename = load_video(path, len=400)
    result, metrics, runtime = _run(video, frames, filename, "../../data/output/cotracker")
    print(metrics)
    print(runtime)

def run(config:dict) -> dict:
    path = config["data"]["path"]
    output_path = config["run"]["artifacts_dir"]
    filtered = config.get("gaussian_filtered", False)
    visibility = config["run"].get("visibility", False)
    diff_warp = config["run"].get("diff_warp", False)
    grid_size = config["run"].get("grid_size", 15)
    video, frames, filename = load_video(path, gaussian_filtered=filtered)
    if config.get("template_strategy", None) == "computed":
        template_index = find_highest_correlation(frames)
    else:
        template_index = 0
    warped, metrics, runtime = _run(video, frames, filename, output_path,
                                    template_index, grid_size, diff_warp, visibility)
    ssim_list = metrics["ssims"]
    mse_list = metrics["mse_list"]
    crispness_improvement = metrics["crispness_improvement"]
    metrics = {
        "per_frame": {
            "ssim": ssim_list,
            "mse": mse_list
        },
        "summary": {
            "mse_mean": float(np.mean(mse_list)),
            "mse_std": float(np.std(mse_list)),
            "crispness_improvement": crispness_improvement
        }
    }
    result = {"runtime_s": runtime,
              "metrics": metrics,
              "artifacts": {
                  "output_path": f"{output_path}/{filename}.mp4",
              }}
    return result


def _run(video, frames, filename, output_path, template_idx=0,
         grid_size=15, diff_warp=False, visibility=False ):
    model = torch.hub.load("facebookresearch/co-tracker",
                           "cotracker3_offline").to(device)
    video.to(device)
    model.model.model_resolution = video.shape[3:]
    start_time = time.time()
    pred_tracks, pred_visibility = model(video=video,
                                         grid_size=grid_size,
                                         grid_query_frame=template_idx,
                                         backward_tracking=True)
    # save video with tracked points
    vis = Visualizer(save_dir=output_path)
    vis.visualize(video, pred_tracks, pred_visibility)

    grid_pts = get_points_on_a_grid(grid_size, model.model.model_resolution)

    result = warping(pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy(),
                     np.expand_dims(frames, axis=-1), diff_warp, visibility)
    end_time = time.time()
    save_and_display_video(np.array(result), f'{output_path}/{filename}.mp4')
    metrics = evaluate(result, frames, frames[template_idx])
    return result, metrics, end_time - start_time

def warping(predicted_tracks, predicted_visibility, frames, diff_warp = False, visibility=False):
    """
    Warps the frames of a video based on predicted tracks.
    args:
        predicted_tracks (torch.Tensor): Predicted tracks of shape (B, T, G, D).
        frames (torch.Tensor): Video frames of shape (T_orig, H, W, C).
    returns:
        warpeds (list): List of warped frames.
    """
    T_orig, H, W, C = frames.shape
    B, T, G, D = predicted_tracks.shape
    grid_size = int(math.sqrt(G))
    velocity = predicted_tracks[0].reshape(T, grid_size, grid_size,
                                           2)  # (24, 32, 32, 2)
    real_velocity = velocity - velocity[0]  # (24, 32, 32, 2)
    if visibility:
        visibility = predicted_visibility[0].reshape(T, grid_size, grid_size)
        vis_mask = (visibility > 0)
        # zero out points that cotracker deems unreliable
        real_velocity *= vis_mask[..., None]
    v = real_velocity.transpose(0, 3, 1, 2)  # (24, 2, 32, 32)
    vp = zoom(v, (1, 1, W / grid_size, H / grid_size))  # (24, 2, 256, 256)

    warpeds = [frames[0][..., 0]]

    for i in range(1, T_orig):
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        phi = np.diff(vp, axis=0)[0:i].sum(0)
        if diff_warp:
            phi = make_monotonic(phi)   #TODO: investigate if this enforces the displacement field to be affine
        grid_x += phi[0].T
        grid_y += phi[1].T

        warped = map_coordinates(frames[i][..., 0].astype(np.float32), [grid_y, grid_x],
                                 order=3, mode='nearest')
        warpeds.append(warped)

    return np.array(warpeds)

def make_monotonic(phi, eps=1e-3):
    """
    Args:
         phi: (2, H, W) displacement field
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


if __name__ == '__main__':
    main()