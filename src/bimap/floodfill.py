"""functions to get ROIs from pixel-wise correlation across frames"""

import matplotlib.pyplot as plt
import numpy as np
from cotracker.utils.visualizer import read_video_from_path

example_path = "../../data/output/ants/method_sweep/strong/v7/run_418fb3ec/artifacts/3czi.mp4"

def main():
    video = read_video_from_path(example_path).squeeze()
    frames = np.array([frame for frame in video], dtype=np.float32)
    corrs = calculate_all_corrs(video)
    rois = build_rois(corrs, 0.95)
    roi_brightness = calculate_roi_brightness_over_time(frames, rois)
    plt.figure(figsize=(10, 6))

    for roi, brightness_curve in roi_brightness.items():
        plt.plot(brightness_curve, label=f"ROI {roi}")

    plt.xlabel("Time")
    plt.ylabel("Average Brightness")
    plt.title("ROI Brightness Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_neighbor_coords(image: np.ndarray, x: int, y: int) -> list[tuple[int, int]]:
    """Calculate the coordinates of all neighboring pixels"""
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


def get_corr_with_neighbors(image_stack: np.ndarray, x: int, y:int) -> float:
    """Calculate the mean correlation of given pixel with its neighbors.

    :param image_stack: (N, H, W) array of N timepoints of HxW images
    :param x: x coordinate of the pixel
    :param y: y coordinate of the pixel
    """
    neighbor_coords = get_neighbor_coords(image_stack[0], x, y)
    pixel_fl_time_series = image_stack[:, y, x]
    neighbors_fl_time_series = np.stack([image_stack[:, ny, nx] for nx, ny in
                                         neighbor_coords], axis=1)
    neighbors_mean_time_series = neighbors_fl_time_series.mean(axis=1)
    return float(np.corrcoef(pixel_fl_time_series, neighbors_mean_time_series)[0, 1])


def calculate_all_corrs(image_stack: np.ndarray) -> np.ndarray:
    """Calculate all individual correlations for each pixel across frames."""
    H, W = image_stack.shape[1], image_stack.shape[2]
    corrs = np.zeros_like(image_stack[0])
    for x in range(W):
        for y in range(H):
            corr = get_corr_with_neighbors(image_stack, x, y)
            corrs[y, x] = corr
    return corrs


def build_roi_recursive(corrs: np.ndarray, center: tuple, threshold: float,
                        labels: np.ndarray, label_id: int) -> set:
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


def build_roi(corrs, threshold, visited):
    """Helper function to build a single ROI"""
    masked_corrs = corrs.copy()
    for y, x in visited:
        masked_corrs[y, x] = -np.inf
    roi = set()
    center = np.unravel_index(np.argmax(masked_corrs, axis=None), corrs.shape)
    roi.update(build_roi_recursive(corrs, center, threshold, visited))
    return roi

def build_rois(corrs: np.ndarray, threshold=0.5) -> np.array:
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
    """:param image_stack: (T, H, W) array of images over time
    :param labels: (H, W) array with ROI labels (1, 2, ...) and 0/-1 for non-ROI
    :return: Dictionary mapping ROI label -> (T,) array of average brightness over time
    """
    roi_brightness = {}
    T = image_stack.shape[0]
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label <= 0:
            continue  # Skip background and visited non-ROI pixels

        # Find pixel indices belonging to the ROI
        roi_mask = labels == label
        roi_pixels = image_stack[:, roi_mask]  # Shape: (T, N_pixels)

        # Compute average over pixels for each timepoint
        roi_mean = roi_pixels.mean(axis=1)  # Shape: (T,)
        roi_brightness[label] = roi_mean

    return roi_brightness

if __name__ == "__main__":
    main()
