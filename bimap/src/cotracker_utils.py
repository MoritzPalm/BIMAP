import math

import numpy as np
from scipy.ndimage import zoom, map_coordinates
import cv2

def warping(predicted_tracks, frames):
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
    v = real_velocity.transpose(0, 3, 1, 2)  # (24, 2, 32, 32)
    vp = zoom(v, (1, 1, W / grid_size, H / grid_size))  # (24, 2, 256, 256)

    warpeds = [frames[0][..., 0]]

    for i in range(1, T_orig):
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        phi = np.diff(vp, axis=0)[0:i].sum(0)
        grid_x += phi[0]
        grid_y += phi[1]

        warped = map_coordinates(frames[i][..., 0].astype(np.float32), [grid_y, grid_x],
                                 order=3, mode='nearest')
        warpeds.append(warped)

    return np.array(warpeds)


def save_and_display_video(array, filename='output.mp4', fps=30):
    num_frames, height, width = array.shape

    # Normalize and convert to uint8 if needed
    if array.dtype != np.uint8:
        array_min = array.min()
        array_max = array.max()
        array = 255 * (array - array_min) / (array_max - array_min + 1e-8)
        array = np.clip(array, 0, 255).astype(np.uint8)

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)

    for i in range(num_frames):
        frame = array[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # <- This line is critical
        out.write(frame)

    out.release()
    print(f"Video saved to {filename}")