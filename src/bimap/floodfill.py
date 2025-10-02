import matplotlib.pyplot as plt
import numpy as np
from cotracker.utils.visualizer import read_video_from_path
from scipy.signal import convolve2d
from utils import load_video
from itertools import cycle


example_path = "../../data/output/ants/method_sweep/strong/v7/run_7f6a7e57/artifacts/3czi.mp4"
example_path = "../../data/input/strong_movement/3czi.tif"

def main():
    # Use your original read + squeeze, but guard single-frame videos to keep (T, H, W)
    frames, _ = load_video(example_path)


    # Compute correlations and ROIs using your original approach
    corrs = corr_with_neighbors_all(frames)
    rois = build_rois(corrs, 0.90)

    # --- NEW: visualize ROIs over the mean image (or pick a frame) ---
    plot_rois_on_image(frames, rois, base="mean")  # "mean" | "max" | "frame:<idx>"

    # ROI brightness (fixed boolean indexing)
    roi_brightness = calculate_roi_brightness_over_time(frames, rois)

    # Plot brightness over time
    plt.figure(figsize=(10, 6))
    for roi, brightness_curve in roi_brightness.items():
        plt.plot(brightness_curve, label=f"ROI {roi}")
    plt.xlabel("Time (frames)")
    plt.ylabel("Average Brightness")
    plt.title("ROI Brightness Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def neighbor_mean_stack(X: np.ndarray) -> np.ndarray:
    """
    X: (N, H, W) float array
    returns M: (N, H, W) neighbor mean for each frame, excluding the center pixel.
    Edge pixels use the mean over available in-bounds neighbors (3/5/8).
    """
    N, H, W = X.shape
    K = np.array([[1,1,1],
                  [1,0,1],
                  [1,1,1]], dtype=X.dtype)
    # counts of valid neighbors are the same for all frames; compute once
    ones = np.ones((H, W), dtype=X.dtype)
    denom = convolve2d(ones, K, mode='same', boundary='fill', fillvalue=0)  # 8 interior, 5 edges, 3 corners

    M = np.empty_like(X)
    for t in range(N):
        s = convolve2d(X[t], K, mode='same', boundary='fill', fillvalue=0)
        M[t] = s / denom
    return M

def corr_with_neighbors_all(X: np.ndarray) -> np.ndarray:
    """
    X: (N, H, W) — N timepoints of HxW images
    returns corrs: (H, W) Pearson correlation between each pixel's time series
                   and the mean of its neighbors' time series.
    """
    # 1) neighbor mean per frame (N,H,W)
    M = neighbor_mean_stack(X)

    N = X.shape[0]
    # 2) Compute timewise statistics without materializing huge intermediates
    sum_x  = X.sum(axis=0)                   # (H,W)
    sum_m  = M.sum(axis=0)                   # (H,W)
    sum_x2 = (X*X).sum(axis=0)               # (H,W)
    sum_m2 = (M*M).sum(axis=0)               # (H,W)
    sum_xm = (X*M).sum(axis=0)               # (H,W)

    # 3) Covariance and variance over time
    # cov(x,m) = E[xm] - E[x]E[m]
    cov_xm = sum_xm - (sum_x * sum_m) / N
    var_x  = sum_x2 - (sum_x * sum_x) / N
    var_m  = sum_m2 - (sum_m * sum_m) / N

    # 4) Pearson correlation with safe divide
    denom = np.sqrt(var_x * var_m)
    corrs = np.zeros_like(denom, dtype=X.dtype)
    mask = denom > 0
    corrs[mask] = (cov_xm[mask] / denom[mask]).astype(X.dtype)
    return corrs

def get_neighbor_coords(image: np.ndarray, x: int, y: int) -> list[tuple[int, int]]:
    H, W = image.shape[0], image.shape[1]
    x_neighbors = [-1, 0, 1]
    y_neighbors = [-1, 0, 1]
    neighbor_coords = []
    for dx in x_neighbors:
        for dy in y_neighbors:

            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                if not (dx == 0 and dy == 0):
                    neighbor_coords.append((nx, ny))
    return neighbor_coords


def build_roi_recursive(corrs: np.ndarray, center: tuple, threshold: float,
                        labels: np.ndarray, label_id: int) -> set:
    # Unchanged logic from your original implementation
    y, x = center
    if labels[y, x] != -1 or corrs[y, x] < threshold:
        return set()

    roi = {center}
    labels[y, x] = label_id  # Temporarily mark, to prevent revisits

    neighbors = get_neighbor_coords(corrs, x, y)
    for nx, ny in neighbors:
        if labels[ny, nx] == -1 and corrs[ny, nx] >= threshold:
            roi.update(build_roi_recursive(corrs, (ny, nx), threshold, labels, label_id))
        elif labels[ny, nx] == -1:
            labels[ny, nx] = 0  # Visited but not part of ROI
    return roi


def build_rois(corrs: np.ndarray, threshold=0.5) -> np.ndarray:
    # Unchanged logic from your original implementation (kept intact)
    labels = np.full_like(corrs, fill_value=-1, dtype=int)
    label_id = 1
    H, W = corrs.shape
    for y in range(H):
        for x in range(W):
            if labels[y, x] == -1 and corrs[y, x] >= threshold:
                roi = build_roi_recursive(corrs, (y, x), threshold, labels, label_id)
                if roi:
                    for ry, rx in roi:
                        labels[ry, rx] = label_id
                    label_id += 1
                else:
                    labels[y, x] = 0  # Mark as visited but not part of ROI
            elif labels[y, x] == -1:
                labels[y, x] = 0
    return labels


def calculate_roi_brightness_over_time(image_stack: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """
    FIXED: boolean indexing across (T, H, W) using flattened mask.
    :param image_stack: (T, H, W)
    :param labels: (H, W) with ROI labels (1, 2, ...) and 0/-1 for non-ROI
    """
    roi_brightness: dict[int, np.ndarray] = {}
    T, H, W = image_stack.shape
    flat_video = image_stack.reshape(T, H * W)
    flat_labels = labels.reshape(H * W)

    for label in np.unique(flat_labels):
        if label <= 0:
            continue
        mask = (flat_labels == label)
        if not np.any(mask):
            continue
        roi_mean = flat_video[:, mask].mean(axis=1)  # (T,)
        roi_brightness[int(label)] = roi_mean

    return roi_brightness

def plot_rois_on_image(frames: np.ndarray, labels: np.ndarray, base: str = "mean") -> None:
    """
    Show ROI areas as colored overlays on top of the base image.
    No labels or numbers, just filled ROI regions.
    """
    T, H, W = frames.shape

    # ----- choose base image -----
    if base == "mean":
        base_img, title_suffix = frames.mean(axis=0), "Mean Image"
    elif base == "max":
        base_img, title_suffix = frames.max(axis=0), "Max-Projection Image"
    elif base.startswith("frame:"):
        try:
            idx = int(base.split(":")[1])
        except Exception:
            idx = 0
        idx = max(0, min(T - 1, idx))
        base_img, title_suffix = frames[idx], f"Frame {idx}"
    else:
        base_img, title_suffix = frames.mean(axis=0), "Mean Image"

    plt.figure(figsize=(8, 8))
    plt.imshow(base_img, cmap="gray", interpolation="nearest")

    # color cycle for ROI fills
    cmap = plt.get_cmap("tab20")
    colors = cycle([cmap(i) for i in range(cmap.N)])

    unique_labels = [int(v) for v in np.unique(labels) if v > 0]
    for lab, color in zip(unique_labels, colors):
        mask = (labels == lab)

        # semi-transparent overlay
        overlay = np.zeros((H, W, 4))
        overlay[..., :3] = color[:3]
        overlay[..., 3] = 0.75 * mask  # alpha only on ROI pixels
        plt.imshow(overlay, interpolation="nearest")

    plt.title(f"ROIs over {title_suffix}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
