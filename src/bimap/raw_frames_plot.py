#!/usr/bin/env python3
"""
Display up to 40 raw video frames in a 5x8 grid with no gaps, each tile showing:
- per-frame 1/99% contrast
- a bottom-right scalebar (same styling as your working script)
- a small frame index badge (top-left)

Loads TIFF via tifffile and MP4/AVI via OpenCV, converts to grayscale float32 in [0,1].
"""

from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import tifffile, cv2
from matplotlib import patheffects
from matplotlib.patches import Rectangle
from skimage.util import img_as_float32


# --------------------- video loading ---------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[-1] == 1:
            return img[..., 0]
        w = np.array([0.2989, 0.5870, 0.1140], dtype=img.dtype)
        return np.tensordot(img[..., :3], w, axes=([-1], [0]))
    raise ValueError(f"Unexpected image shape: {img.shape}")

def load_all_frames_thw(path: str) -> np.ndarray:
    """Return (T,H,W) float32 in [0,1]."""
    path = os.fspath(path)
    if path.lower().endswith((".tif", ".tiff")):
        with tifffile.TiffFile(path) as tf:
            frames = [img_as_float32(_to_gray(pg.asarray())) for pg in tf.pages]
        if not frames:
            raise RuntimeError(f"No pages in TIFF: {path}")
        frames = frames[:400]
        return np.stack(frames, axis=0)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(img_as_float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {path}")
    frames = frames[:400]
    return np.stack(frames, axis=0)


# --------------------- scalebar (same formatting as your working script) ---------------------

def _choose_bar_um(img_w_px: int, px_um: float, target_frac: float = 0.15) -> float:
    nice = [1, 2, 5, 10, 20, 50, 100, 200]
    desired_um = img_w_px * px_um * target_frac
    return min(nice, key=lambda x: abs(x - desired_um))

def _draw_scale_bar(ax, img_shape, px_um: float, bar_um: float | None = None):
    """Draw bottom-right scale bar + label fully inside axes."""
    H, W = img_shape
    if bar_um is None:
        bar_um = _choose_bar_um(W, px_um)
    bar_px = max(1, int(round(bar_um / px_um)))

    margin_y = int(0.04 * H)
    margin_x = int(0.04 * W)
    thickness = max(4, int(0.012 * min(H, W)))  # thicker white fill

    x0 = max(1, W - margin_x - bar_px)
    y0 = max(1, H - margin_y - thickness)

    rect = Rectangle(
        (x0, y0), bar_px, thickness,
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95, zorder=10,
    )
    ax.add_patch(rect)

    y_text = max(1, y0 - 3 * thickness)
    txt = f"{bar_um:g} µm"
    text = ax.text(
        x0 + bar_px / 2, y_text, txt,
        ha="center", va="bottom",
        color="white", fontsize=9, zorder=11,
    )
    text.set_path_effects([
        patheffects.Stroke(linewidth=0.5, foreground="black"),
        patheffects.Normal(),
    ])


# --------------------- helpers ---------------------

def pick_frame_indices(T: int, n: int) -> List[int]:
    """Evenly spaced indices in [0, T-1], length <= n."""
    if T <= 0:
        return []
    if n >= T:
        return list(range(T))
    pos = np.linspace(0, T, num=n, dtype=int, endpoint=False)
    out, seen = [], set()
    for p in pos:
        q = int(p)
        if q not in seen:
            out.append(q); seen.add(q)
    i = 0
    while len(out) < n and i < T:
        if i not in seen:
            out.append(i); seen.add(i)
        i += 1
    out.sort()
    return out[:n]


# --------------------- main ---------------------

def parse_args():
    ap = argparse.ArgumentParser(description="5x8 grid of up to 40 raw frames with scalebar and frame index.")
    ap.add_argument("--raw_video", required=True, help="Path to raw video (TIFF stack or MP4/AVI).")
    ap.add_argument("--pixel_size_um", type=float, required=True, help="Pixel size in microns (isotropic).")
    ap.add_argument("--bar_um", type=float, default=None, help="Fixed scalebar length in microns (optional).")

    ap.add_argument("--frames", type=int, default=40, help="Max frames to plot (<= 40).")
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--panel_width_in", type=float, default=2.0, help="Tile width in inches.")
    ap.add_argument("--row_gap", type=float, default=0.0, help="Normalized gap between rows (0 = none).")
    ap.add_argument("--col_gap", type=float, default=0.0, help="Normalized gap between cols (0 = none).")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--cmap", default="gray")
    ap.add_argument("--outdir", default="raw_grid_5x8")
    return ap.parse_args()

def main():
    args = parse_args()

    vid = load_all_frames_thw(args.raw_video)  # (T,H,W) float32
    T, H, W = vid.shape
    rows, cols = args.rows, args.cols

    n = min(args.frames, rows * cols, T)
    idxs = pick_frame_indices(T, n)

    # Figure size (tiles all same size). Height derived from image aspect to keep visual proportions nice.
    aspect = H / W
    tile_w_in = args.panel_width_in
    tile_h_in = tile_w_in * aspect
    fig_w_in = cols * tile_w_in + (cols - 1) * (args.col_gap * tile_w_in)
    fig_h_in = rows * tile_h_in + (rows - 1) * (args.row_gap * tile_h_in)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Normalized tile sizes so cols*tile + (cols-1)*gap = 1
    tile_w_norm = (1.0 - (cols - 1) * args.col_gap) / cols if cols > 1 else 1.0
    tile_h_norm = (1.0 - (rows - 1) * args.row_gap) / rows if rows > 1 else 1.0

    # Draw
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        x0 = c * (tile_w_norm + args.col_gap)
        y0 = 1.0 - (r + 1) * tile_h_norm - r * args.row_gap
        ax = fig.add_axes([x0, y0, tile_w_norm, tile_h_norm])
        ax.set_axis_off()
        if k < n:
            i = idxs[k]
            v1, v99 = np.percentile(vid[i], (1, 99))
            if v99 <= v1:
                v99 = v1 + 1e-6
            ax.imshow(vid[i], cmap=args.cmap, vmin=float(v1), vmax=float(v99),
                      interpolation="nearest", aspect="auto")
            # frame index badge (top-left)
            ax.text(0.02, 0.98, f"{i}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7, color="white",
                    bbox=dict(facecolor="black", edgecolor="none",
                              alpha=0.5, boxstyle="round,pad=0.15"))
            # scalebar
            _draw_scale_bar(ax, vid[i].shape, args.pixel_size_um, args.bar_um)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_pdf = outdir / f"{Path(args.raw_video).stem}_grid_{rows}x{cols}_{n}frames.pdf"
    fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")

if __name__ == "__main__":
    main()
