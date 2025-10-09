#!/usr/bin/env python3
"""
Per-source temporal signed diffs within the SAME video (one column layout):
    default: diff_t = frame_0 - frame_t
    (flip with --temporal-order t-minus-first; change baseline with --baseline-idx)

Frames shown: 50, 100, ..., 400 (clipped to T-1).
One axis-free PDF per source (raw, ants, cotracker, normcorre, lddmms), laid out as a single column.

Behavior / Assumptions:
- Video loading matches your working script:
  * TIFF via tifffile (reads pages)
  * MP4/AVI via OpenCV (reads frames)
  * converts to grayscale and float32 in [0,1] (skimage.img_as_float32)
- Per-method or per-frame p99(|diff|) scaling (NO global scaling).
- Run selection per method via discover_runs -> pick_run -> find_artifact (glob only; NO JSON key).
- Robust spatial resize (cv2 INTER_AREA, NN fallback), NaN-safe diffs.
- Scale bar per panel using the EXACT same formatting as your working script.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt

import cv2
import tifffile
from matplotlib import patheffects
from matplotlib.patches import Rectangle
from skimage.util import img_as_float32

METHODS_DEFAULT = ("ants", "cotracker", "normcorre", "lddmms")
ARTIFACT_EXTS = (".tif", ".tiff", ".mp4", ".avi")


# ---------- run discovery + selection (glob-only artifacts) ----------

def read_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def discover_runs(runs_root: str, method: str, experiment: str, category: str, video_id: str) -> list[str]:
    base = Path(runs_root) / method / experiment / category / video_id
    return sorted([str(p) for p in base.glob("run_*") if p.is_dir()])

def pick_run(run_dirs: list[str], select: str, metric_mode: str = "max") -> str | None:
    if not run_dirs:
        return None
    if select == "latest":
        return max(run_dirs, key=lambda d: os.path.getmtime(d))
    m = re.match(r"^best:(.+)$", select)
    if m:
        key = m.group(1)
        best_dir, best_val = None, None
        for d in run_dirs:
            rj = read_json(os.path.join(d, "result.json")) or {}
            val = rj
            for part in key.split("."):
                val = val.get(part) if isinstance(val, dict) else None
            if val is None:
                continue
            if best_val is None or (metric_mode == "max" and val > best_val) or (metric_mode == "min" and val < best_val):
                best_val, best_dir = val, d
        return best_dir or max(run_dirs, key=lambda d: os.path.getmtime(d))
    return max(run_dirs, key=lambda d: os.path.getmtime(d))

def find_artifact(run_dir: str, artifact_glob: str) -> str | None:
    """Glob-only artifact finder; ignores a file literally named 'video.mp4'."""
    patterns = [p.strip() for p in artifact_glob.split(",") if p.strip()]
    for pat in patterns:
        for cand in glob.glob(os.path.join(run_dir, pat)):
            name = os.path.basename(cand).lower()
            if name == "video.mp4":
                continue
            if cand.lower().endswith(ARTIFACT_EXTS):
                return cand
    return None


# ---------- loading (same style as working script, but for ALL frames) ----------

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert (H,W) or (H,W,C) to (H,W) grayscale like the working script."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        C = img.shape[-1]
        if C == 1:
            return img[..., 0]
        w = np.array([0.2989, 0.5870, 0.1140], dtype=img.dtype)
        return np.tensordot(img[..., :3], w, axes=([-1], [0]))
    raise ValueError(f"Unexpected image shape: {img.shape}")

def load_all_frames_thw(path: str, max_len: int = 400) -> np.ndarray:
    """
    Load an entire video as (T,H,W) float32 in [0,1], using:
      - tifffile for TIFF stacks
      - OpenCV for MP4/AVI
    Mirrors the working script’s grayscale + img_as_float32 path.
    Truncates to at most max_len frames (default 400).
    """
    path = os.fspath(path)
    if path.lower().endswith((".tif", ".tiff")):
        with tifffile.TiffFile(path) as tf:
            T = len(tf.pages)
            T_use = min(T, max_len) if max_len is not None and max_len > 0 else T
            frames = []
            for k in range(T_use):
                arr = tf.pages[k].asarray()
                g = _to_gray(arr)
                frames.append(img_as_float32(g))
        return np.stack(frames, axis=0)

    # Video via OpenCV
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    frames: List[np.ndarray] = []
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(img_as_float32(gray8))
            count += 1
            if max_len is not None and max_len > 0 and count >= max_len:
                break
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from video: {path}")
    return np.stack(frames, axis=0)


# ---------- utils (indices, resize, scalebar, saving) ----------

def make_indices(T: int, step: int = 50, start: int = 50, max_idx: int = 400) -> List[int]:
    """Return [start, start+step, ...] up to max_idx and < T."""
    last = min(T - 1, max_idx)
    if start < 0:
        start = 0
    if last < start:
        return []
    idxs = list(range(start, last + 1, step))
    return [i for i in idxs if 0 <= i < T]

def safe_resize_like(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if img.shape == ref.shape:
        return img
    try:
        return cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    except Exception:
        # simple NN fallback
        y_scale = ref.shape[0] / img.shape[0]
        x_scale = ref.shape[1] / img.shape[1]
        ys = (np.arange(ref.shape[0]) / y_scale).astype(int).clip(0, img.shape[0] - 1)
        xs = (np.arange(ref.shape[1]) / x_scale).astype(int).clip(0, img.shape[1] - 1)
        return img[ys[:, None], xs[None, :]]

# --- scalebar: EXACTLY as in the working script ---
def _choose_bar_um(img_w_px: int, px_um: float, target_frac: float = 0.15) -> float:
    nice = [1, 2, 5, 10, 20, 50, 100, 200]
    desired_um = img_w_px * px_um * target_frac
    return min(nice, key=lambda x: abs(x - desired_um))

def _draw_scale_bar(ax, img_shape, px_um: float, bar_um: float | None = None):
    """Draw a bottom-right scale bar + label fully INSIDE the axes.
    White bar thicker; thin black outline and text halo to match.
    """
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
        linewidth=0.5,   # thin outline
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
        patheffects.Stroke(linewidth=0.5, foreground="black"),  # thin halo
        patheffects.Normal(),
    ])

def save_diff_column_pdf(out_pdf: str,
                         diffs: List[np.ndarray],
                         indices: Sequence[int],
                         title: str,
                         vmaxs: Sequence[float],
                         cmap: str,
                         dpi: int,
                         panel_size_in: float,
                         pixel_size_um: float,
                         bar_um: float | None):
    """
    Render a single column of panels (one diff per row), with a scalebar per panel.
    `vmaxs` is per-frame vmax (already computed outside).
    """
    n = len(diffs)
    if n == 0:
        return
    fig_w = panel_size_in
    fig_h = n * panel_size_in + 0.35
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, y=0.995, fontsize=9)

    top_pad = 0.10  # fraction for title area
    for k, (img, idx, vmax) in enumerate(zip(diffs, indices, vmaxs, strict=False)):
        # Full-width single column; stack from top to bottom
        ax = fig.add_axes([
            0.0,
            0.02 + (n - 1 - k) / n * (1 - 0.02 - top_pad),
            1.0,
            (1 - 0.02 - top_pad) / n
        ])
        ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=+vmax, interpolation="nearest")
        ax.axis("off")
        # Frame index badge
        ax.text(0.02, 0.98, f"{idx}", transform=ax.transAxes,
                ha="left", va="top", fontsize=7, color="white",
                bbox=dict(facecolor="black", edgecolor="none",
                          alpha=0.5, boxstyle="round,pad=0.15"))
        # Scale bar (exact formatting)
        _draw_scale_bar(ax, img.shape, pixel_size_um, bar_um)

    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Per-source temporal signed diffs (first vs t) in one column, with scalebars.")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--category", required=True, choices=["low", "strong"])
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--raw-video", required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS_DEFAULT))
    ap.add_argument("--outdir", default="temporal_diffs_first_vs_t_column")
    ap.add_argument("--frames-step", type=int, default=50)
    ap.add_argument("--frames-start", type=int, default=50, help="Start index for plotting (default 50).")
    ap.add_argument("--frames-max", type=int, default=400)
    ap.add_argument("--baseline-idx", type=int, default=0, help="Baseline frame index (0 means 'frame 1').")
    ap.add_argument("--cmap", default="seismic")  # zero-centered diverging
    ap.add_argument("--scaling", choices=["permethod", "perframe"], default="permethod")  # NO global scaling
    ap.add_argument("--temporal-order", choices=["first-minus-t", "t-minus-first"], default="first-minus-t")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--panel-size-in", type=float, default=2.0)
    ap.add_argument("--zero_mean", action="store_true",
                    help="Subtract per-frame mean from the temporal diff (visual-only DC bias removal).")

    # Scale bar (same formatting as working script)
    ap.add_argument("--pixel-size-um", type=float, required=True,
                    help="Pixel size in microns (width == height).")
    ap.add_argument("--bar-um", type=float, default=None,
                    help="Fixed scalebar length in microns; if omitted, auto-choose.")

    # Run selection & artifact picking (glob-only)
    ap.add_argument("--select", default="latest",
                    help="Which run to use per method: 'latest' or 'best:<json.key>' (e.g., best:metrics.ssim)")
    ap.add_argument("--metric-mode", choices=["max", "min"], default="max",
                    help="If using 'best:<json.key>', whether higher (max) or lower (min) is better")
    ap.add_argument("--artifact-glob", default="artifacts/*.tif,artifacts/*.tiff,artifacts/*.mp4,artifacts/*.avi",
                    help="Comma-separated globs (relative to run dir) to find the artifact. JSON keys are NOT used.")
    return ap.parse_args()


# ---------- main ----------

def main():
    args = parse_args()

    out_root = Path(args.outdir) / args.experiment / args.category / args.video_id
    out_root.mkdir(parents=True, exist_ok=True)

    # Load RAW once -> (T,H,W) float32 in [0,1]; truncate to <= frames_max (<= 400)
    raw = load_all_frames_thw(args.raw_video, max_len=args.frames_max)
    T_raw, H, W = raw.shape

    # Build the list of target indices (50, 100, ..., <= frames_max and < T)
    indices = make_indices(T_raw, step=args.frames_step, start=args.frames_start, max_idx=args.frames_max)

    # Include RAW as a source too
    sources: Dict[str, np.ndarray] = {"raw": raw}

    # Load each method via run selection
    for method in args.methods:
        run_dirs = discover_runs(args.runs_root, method, args.experiment, args.category, args.video_id)
        chosen = pick_run(run_dirs, args.select, args.metric_mode)
        if not chosen:
            print(f"[WARN] No runs for '{method}'. Skipping.")
            continue
        art = find_artifact(chosen, args.artifact_glob)
        if not art:
            print(f"[WARN] No artifact found for '{method}' in {os.path.basename(chosen)}. Skipping.")
            continue
        print(f"{method}: using run {os.path.basename(chosen)} -> artifact: {os.path.relpath(art, chosen)}")

        mv = load_all_frames_thw(art, max_len=args.frames_max)  # (T,H,W) float32 in [0,1]
        # Spatial align to RAW (if needed)
        if mv.shape[1:] != (H, W):
            mv_resized = np.empty((mv.shape[0], H, W), dtype=np.float32)
            for t in range(mv.shape[0]):
                mv_resized[t] = safe_resize_like(mv[t], raw[0])
            mv = mv_resized
        sources[method] = mv

    # Render one PDF per source (temporal diffs inside each source)
    for src, vid in sources.items():
        T = vid.shape[0]
        if args.baseline_idx < 0 or args.baseline_idx >= T:
            print(f"[WARN] '{src}': baseline index {args.baseline_idx} is out of bounds (T={T}). Skipping.")
            continue

        idx_valid = [i for i in indices if 0 <= i < T and i != args.baseline_idx]
        if not idx_valid:
            print(f"[WARN] '{src}': no valid frames to plot. Skipping.")
            continue

        base = np.nan_to_num(vid[args.baseline_idx], nan=0.0)
        diffs: List[np.ndarray] = []
        for i in idx_valid:
            cur = np.nan_to_num(vid[i], nan=0.0)
            d = (base - cur) if args.temporal_order == "first-minus-t" else (cur - base)
            if args.zero_mean:
                d = d - float(np.mean(d))
            diffs.append(d)

        # Decide scaling
        if args.scaling == "perframe":
            vmaxs = []
            for img in diffs:
                vmax = float(np.percentile(np.abs(img), 99.0))
                vmaxs.append(max(vmax, 1e-6))
        else:  # permethod
            arr = np.abs(np.stack(diffs, axis=0))
            vmax = float(np.percentile(arr, 99.0))
            vmax = max(vmax, 1e-6)
            vmaxs = [vmax] * len(diffs)

        out_pdf = out_root / f"{src}_temporal_diffs_first_vs_t_column.pdf"
        save_diff_column_pdf(
            str(out_pdf),
            diffs,
            idx_valid,
            title=(f"{src}: signed temporal diffs "
                   f"({ 'frame_0 − frame_t' if args.temporal_order=='first-minus-t' else 'frame_t − frame_0' }) "
                   f"(baseline={args.baseline_idx}, {'per-frame' if args.scaling=='perframe' else 'per-method'} scaling)"),
            vmaxs=vmaxs,
            cmap=args.cmap,
            dpi=args.dpi,
            panel_size_in=args.panel_size_in,
            pixel_size_um=args.pixel_size_um,
            bar_um=args.bar_um,
        )
        print(f"[OK] {out_pdf}")

    print(f"Done. Output root: {out_root}")


if __name__ == "__main__":
    main()
