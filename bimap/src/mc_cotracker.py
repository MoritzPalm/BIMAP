import math
import time

import numpy as np
import torch
from scipy.ndimage import zoom, map_coordinates
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.models.core.model_utils import get_points_on_a_grid

from utils import load_video, save_and_display_video, evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    path = "../../data/input/strong_movement/b5czi.tif"
    video, frames, filename = load_video(path, len=400)
    result, metrics, runtime = run_cotracker(video, frames, filename)
    print(metrics)
    print(runtime)


def run_cotracker(video, frames, filename, template_idx=0, grid_size=15):
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
    vis = Visualizer(save_dir="../../output/cotracker/visualizations")
    vis.visualize(video, pred_tracks, pred_visibility)

    grid_pts = get_points_on_a_grid(grid_size, model.model.model_resolution)

    result = warping(pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy(),
                     np.expand_dims(frames, axis=-1))
    end_time = time.time()
    save_and_display_video(np.array(result), f'../../data/output/cotracker'
                                             f'/{filename}_visibility.mp4')
    metrics = evaluate(result, frames, frames[0])
    return result, metrics, end_time - start_time

def warping(predicted_tracks, predicted_visibility, frames):
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
    visibility = predicted_visibility[0].reshape(T, grid_size, grid_size)
    vis_mask = (visibility > 0)
    velocity = predicted_tracks[0].reshape(T, grid_size, grid_size,
                                           2)  # (24, 32, 32, 2)
    real_velocity = velocity - velocity[0]  # (24, 32, 32, 2)
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
        #phi = make_monotonic(phi)   #TODO: investigate if this enforces the displacement field to be affine
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