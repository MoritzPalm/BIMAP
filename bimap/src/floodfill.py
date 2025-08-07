"""functions to get ROIs from pixel-wise correlation across frames"""

import numpy as np


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


def build_roi_recursive(corrs: np.ndarray, center: tuple, threshold=0.5,
                        visited: set = None) ->set:
    """Recursive building of a single ROI"""
    if visited is None:
        visited = set()
    if center in visited or corrs[center] < threshold:
        return set()

    visited.add(center)
    roi = {center}
    neighbors = get_neighbor_coords(corrs, center[1], center[0])
    for x, y in neighbors:
        if corrs[y, x] > threshold:
            roi.update(build_roi_recursive(corrs, (y, x), threshold, visited))
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


def build_rois(corrs: np.ndarray, threshold=0.5) -> list:
    corrs_cpy = corrs.copy()
    rois = []
    visited = set()
    for i in range(30000):  # building three rois
        roi = build_roi(corrs_cpy, threshold, visited)
        if roi:
            rois.append(roi)
            visited.update(roi)
    return rois
