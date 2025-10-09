#!/usr/bin/env python3
"""Two-frame visualization from runs/ with per-source outputs and tight summaries.

Per source (raw + methods):
  - frame_000.npy / frame_000.png
  - frame_<N>.npy / frame_<N>.png
  - <source>_diff_000_minus_N.npy
  - <source>_diff_000_minus_N_signed.(png/svg/pdf)
  - <source>_diff_000_minus_N_abs.(png/svg/pdf)

Summaries (SVG):
  - summary_frameN.svg     # images at frame N (or frame 0 via --summary-frame 0)
  - summary_diffs_signed_<mode>_<scale>.svg  # signed diffs with good scaling

Summary layout:
- Methods in a 2x2 grid (left→right, top→bottom): ANTs, CoTracker, LDDMMs, NoRMCorre
- RAW frame below, centered horizontally, same size as each method panel.

Notes:
- OpenCV backend for MP4/AVI; tifffile for TIFFs.
- Any artifact whose basename is literally "video.mp4" is ignored.
- All PNGs include a scale bar + label (µm). Use --bar-um to fix the displayed length.

"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib import patheffects
from matplotlib.patches import Rectangle
from skimage.util import img_as_float32

# -------------------------- runs discovery -------------------------- #

def read_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def discover_runs(runs_root: str, group: str, experiment: str, category: str, vid: str) -> list[str]:
    base = os.path.join(runs_root, group, experiment, category, vid)
    return sorted(glob.glob(os.path.join(base, "run_*")))

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

def find_artifact(run_dir: str, artifact_glob: str, artifact_json_key: str | None) -> str | None:
    # Prefer JSON-indicated artifact
    if artifact_json_key:
        for fn in ("result.json", "_child_return.json", "config.json"):
            js = read_json(os.path.join(run_dir, fn))
            if not js:
                continue
            val = js
            for part in artifact_json_key.split("."):
                val = val.get(part) if isinstance(val, dict) else None
            if isinstance(val, str):
                cand = val if os.path.isabs(val) else os.path.join(run_dir, val)
                if os.path.exists(cand) and os.path.basename(cand) != "video.mp4":
                    return cand
    # Fallback: glob
    patterns = [p.strip() for p in artifact_glob.split(",") if p.strip()]
    for pat in patterns:
        for cand in glob.glob(os.path.join(run_dir, pat)):
            if cand.lower().endswith((".tif", ".tiff", ".mp4", ".avi")) and os.path.basename(cand) != "video.mp4":
                return cand
    return None


# -------------------------- two-frame loading -------------------------- #

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        C = img.shape[-1]
        if C == 1:
            return img[..., 0]
        w = np.array([0.2989, 0.5870, 0.1140], dtype=img.dtype)
        return np.tensordot(img[..., :3], w, axes=([-1], [0]))
    raise ValueError(f"Unexpected image shape: {img.shape}")

def essential_indices(T: int, f2: int) -> tuple[int, int]:
    return 0, min(f2, max(0, T - 1))

def load_two_frames(path: str, frame2: int = 399) -> list[np.ndarray]:
    """Return [frame0, frameN] as float32 (H,W). Uses OpenCV for videos; tifffile for TIFFs."""
    path = os.fspath(path)
    if path.lower().endswith((".tif", ".tiff")):
        with tifffile.TiffFile(path) as tf:
            T = len(tf.pages)
            idx0, idxN = essential_indices(T, frame2)
            imgs = [tf.pages[idx0].asarray(), tf.pages[idxN].asarray()]
        return [img_as_float32(_to_gray(x)) for x in imgs]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx0, idxN = essential_indices(T if T > 0 else frame2 + 1, frame2)

    def read_index(k: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {k} from {path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return img_as_float32(gray)

    try:
        f0 = read_index(idx0)
        fN = read_index(idxN)
    finally:
        cap.release()

    return [f0, fN]


# -------------------------- saving helpers (with scale bar) -------------------------- #

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _choose_bar_um(img_w_px: int, px_um: float, target_frac: float = 0.15) -> float:
    nice = [1, 2, 5, 10, 20, 50, 100, 200]
    desired_um = img_w_px * px_um * target_frac
    return min(nice, key=lambda x: abs(x - desired_um))

def _draw_scale_bar(ax, img_shape, px_um: float, bar_um: float | None = None):
    """Draw a bottom-right scale bar + label fully INSIDE the axes."""
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

def save_npy_png_bar(img: np.ndarray, npy_path: str, png_path: str,
                     percentile_vis: tuple[float, float] = (1, 99),
                     *, pixel_size_um: float, bar_um: float | None = None,
                     dpi: int = 600, figsize_in: float = 5.0,
                     save_svg: bool = False, save_pdf: bool = False):
    np.save(npy_path, img)
    vmin = np.percentile(img, percentile_vis[0]); vmax = np.percentile(img, percentile_vis[1])
    if vmax <= vmin: vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(figsize_in, figsize_in))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])

    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")
    _draw_scale_bar(ax, img.shape, pixel_size_um, bar_um)

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.005)
    if save_svg:
        fig.savefig(os.path.splitext(png_path)[0] + ".svg", bbox_inches="tight", pad_inches=0.005)
    if save_pdf:
        fig.savefig(os.path.splitext(png_path)[0] + ".pdf", bbox_inches="tight", pad_inches=0.005)
    plt.close(fig)

def save_diff_images_bar(img0: np.ndarray, imgN: np.ndarray, out_dir: str, prefix: str,
                         *, pixel_size_um: float, bar_um: float | None = None,
                         dpi: int = 600, figsize_in: float = 5.0,
                         save_svg: bool = False, save_pdf: bool = False):
    diff = img0 - imgN
    np.save(os.path.join(out_dir, f"{prefix}_diff_000_minus_N.npy"), diff)

    v = np.percentile(np.abs(diff), 99)
    v = float(v) if v > 0 else 1e-6

    # Signed diff
    fig, ax = plt.subplots(figsize=(figsize_in, figsize_in))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    ax.imshow(diff, cmap="seismic", vmin=-v, vmax=v)
    ax.axis("off")
    _draw_scale_bar(ax, diff.shape, pixel_size_um, bar_um)
    spng = os.path.join(out_dir, f"{prefix}_diff_000_minus_N_signed.png")
    fig.savefig(spng, dpi=dpi, bbox_inches="tight", pad_inches=0.005)
    if save_svg:
        fig.savefig(os.path.splitext(spng)[0] + ".svg", bbox_inches="tight", pad_inches=0.005)
    if save_pdf:
        fig.savefig(os.path.splitext(spng)[0] + ".pdf", bbox_inches="tight", pad_inches=0.005)
    plt.close(fig)

    # Abs diff
    diff_abs = np.abs(diff)
    fig, ax = plt.subplots(figsize=(figsize_in, figsize_in))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    ax.imshow(diff_abs, cmap="magma")
    ax.axis("off")
    _draw_scale_bar(ax, diff_abs.shape, pixel_size_um, bar_um)
    apng = os.path.join(out_dir, f"{prefix}_diff_000_minus_N_abs.png")
    fig.savefig(apng, dpi=dpi, bbox_inches="tight", pad_inches=0.005)
    if save_svg:
        fig.savefig(os.path.splitext(apng)[0] + ".svg", bbox_inches="tight", pad_inches=0.005)
    if save_pdf:
        fig.savefig(os.path.splitext(apng)[0] + ".pdf", bbox_inches="tight", pad_inches=0.005)
    plt.close(fig)


# -------------------------- tight summaries: 2x2 methods + centered RAW -------------------------- #

def _tight_grid_rects_2x2_plus_raw_center(aspect: float,
                                          gutter: float = 0.005,
                                          label_pad_top: float = 0.035,
                                          label_pad_bottom: float = 0.035,
                                          outer_pad: float = 0.01):
    """
    Two rows x two columns for methods, then one RAW panel below, centered horizontally.
    All panels (including RAW) have identical size.
    """
    # Horizontal: 2 columns -> 1 gutter
    avail_w = 1.0 - 2 * outer_pad - gutter
    panel_w = avail_w / 2.0
    panel_h = panel_w * aspect

    # Vertical: methods row1 + gutter + methods row2 + gutter + raw row + top/bottom label pads
    avail_h = 1.0 - 2 * outer_pad - label_pad_top - label_pad_bottom - 2 * gutter
    needed_h = 3 * panel_h + 2 * gutter
    if needed_h > avail_h:
        scale = avail_h / needed_h
        panel_w *= scale
        panel_h *= scale
        # recompute horizontal availability after scaling (for completeness)
        avail_w = 1.0 - 2 * outer_pad - gutter

    # X positions for columns
    x_left  = outer_pad
    x_right = outer_pad + panel_w + gutter

    # Y positions
    # Top row of methods should be visually topmost; place raw at the very bottom
    y_bottom_raw = outer_pad + label_pad_top
    y_mid        = y_bottom_raw + panel_h + gutter
    y_top        = y_mid + panel_h + gutter

    # Rects for methods (2x2)
    rects_methods = [
        [x_left,  y_top, panel_w, panel_h],    # ANTs
        [x_right, y_top, panel_w, panel_h],    # CoTracker
        [x_left,  y_mid, panel_w, panel_h],    # LDDMMs
        [x_right, y_mid, panel_w, panel_h],    # NoRMCorre
    ]

    # RAW centered horizontally with same size as a single panel
    x_center = 0.5 - panel_w / 2.0
    rect_raw = [x_center, y_bottom_raw, panel_w, panel_h]

    return rects_methods, rect_raw, label_pad_top, label_pad_bottom

def _imshow_panel(ax, img: np.ndarray):
    vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
    if vmax <= vmin: vmax = vmin + 1e-6
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")

def save_summary_grid_2x2plusraw_center(images: dict[str, np.ndarray],
                                        out_svg: str,
                                        *, pixel_size_um: float, bar_um: float | None,
                                        figsize_in: float = 7.0,
                                        gutter: float = 0.005,
                                        label_pad_top: float = 0.035,
                                        label_pad_bottom: float = 0.035,
                                        outer_pad: float = 0.01):
    """
    Methods in 2x2 (ANTs, CoTracker, LDDMMs, NoRMCorre), RAW centered below, same size.
    """
    order_methods = [("ants", "ANTs"),
                     ("cotracker", "CoTracker"),
                     ("lddmms", "LDDMMs"),
                     ("normcorre", "NoRMCorre")]
    raw_key = ("raw", "Raw")

    # Aspect from any available panel
    ref = None
    for k, _ in order_methods + [raw_key]:
        if images.get(k) is not None:
            ref = images[k]
            break
    if ref is None:
        raise ValueError("No images available for summary grid.")
    aspect = ref.shape[0] / ref.shape[1]

    fig = plt.figure(figsize=(figsize_in, figsize_in))
    rects_methods, rect_raw, pad_top, pad_bottom = _tight_grid_rects_2x2_plus_raw_center(
        aspect, gutter, label_pad_top, label_pad_bottom, outer_pad
    )

    # Methods (2x2)
    axes_methods = []
    for (key, _), rect in zip(order_methods, rects_methods, strict=False):
        ax = fig.add_axes(rect)
        ax.axis("off")
        img = images.get(key)
        if img is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    color="red", fontsize=12, transform=ax.transAxes)
        else:
            _imshow_panel(ax, img)
            _draw_scale_bar(ax, img.shape, pixel_size_um, bar_um)
        axes_methods.append(ax)

    # RAW (centered, same size)
    ax_raw = fig.add_axes(rect_raw)
    ax_raw.axis("off")
    img_raw = images.get(raw_key[0])
    if img_raw is None:
        ax_raw.text(0.5, 0.5, "missing", ha="center", va="center",
                    color="red", fontsize=12, transform=ax_raw.transAxes)
    else:
        _imshow_panel(ax_raw, img_raw)
        _draw_scale_bar(ax_raw, img_raw.shape, pixel_size_um, bar_um)

    # Labels: above methods, below RAW
    labels_methods = [lbl for _, lbl in order_methods]
    for ax, lbl in zip(axes_methods, labels_methods, strict=False):
        bbox = ax.get_position()
        x = (bbox.x0 + bbox.x1) / 2
        y = bbox.y1 + (label_pad_top * 0.6)
        fig.text(x, y, lbl, ha="center", va="bottom", fontsize=12, color="black")

    bbox = ax_raw.get_position()
    x = (bbox.x0 + bbox.x1) / 2
    y = bbox.y0 - (label_pad_bottom * 0.6)
    fig.text(x, y, raw_key[1], ha="center", va="top", fontsize=12, color="black")

    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

def save_summary_grid_signed_diffs_2x2plusraw_center(frames0: dict[str, np.ndarray],
                                                     framesN: dict[str, np.ndarray],
                                                     raw0: np.ndarray, rawN: np.ndarray,
                                                     out_svg: str,
                                                     *, pixel_size_um: float, bar_um: float | None,
                                                     figsize_in: float = 7.0,
                                                     gutter: float = 0.005,
                                                     label_pad_top: float = 0.035,
                                                     label_pad_bottom: float = 0.035,
                                                     outer_pad: float = 0.01,
                                                     scale: str = "perpanel",
                                                     mode: str = "temporal",
                                                     vmax_fixed: float | None = None):
    """
    Signed diffs in 2x2 for methods (ANTs, CoTracker, LDDMMs, NoRMCorre),
    RAW diff centered below, same size.
    """
    order_methods = [("ants", "ANTs"),
                     ("cotracker", "CoTracker"),
                     ("lddmms", "LDDMMs"),
                     ("normcorre", "NoRMCorre")]

    # Aspect from any available 0-frame
    ref = next((frames0.get(k) for k, _ in [("raw","Raw")] + order_methods if frames0.get(k) is not None), None)
    if ref is None:
        raise ValueError("No frames available for diff summary.")
    aspect = ref.shape[0] / ref.shape[1]

    # Compute diffs & scaling stats
    diffs, p99s = {}, []
    keys_all = ["raw"] + [k for k, _ in order_methods]
    for k in keys_all:
        f0, fN = frames0.get(k), framesN.get(k)
        if f0 is None or fN is None:
            diffs[k] = None
            p99s.append(0.0)
            continue
        if mode == "temporal":
            d = f0 - fN
        elif k == "raw":
            d = fN - rawN
        else:
            d = fN - rawN
        diffs[k] = d
        p99s.append(float(np.percentile(np.abs(d), 99)))

    vmax_global = max(p99s) if p99s else 1e-6
    if vmax_global <= 0:
        vmax_global = 1e-6

    fig = plt.figure(figsize=(figsize_in, figsize_in))
    rects_methods, rect_raw, pad_top, pad_bottom = _tight_grid_rects_2x2_plus_raw_center(
        aspect, gutter, label_pad_top, label_pad_bottom, outer_pad
    )

    # Methods
    axes_methods = []
    for (k, lbl), rect in zip(order_methods, rects_methods, strict=False):
        ax = fig.add_axes(rect)
        ax.axis("off")
        d = diffs.get(k)
        if d is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    color="red", fontsize=12, transform=ax.transAxes)
        else:
            if vmax_fixed is not None:
                vmax = float(vmax_fixed) if vmax_fixed > 0 else 1e-6
            elif scale == "perpanel":
                vmax = float(np.percentile(np.abs(d), 99)) or 1e-6
            else:
                vmax = vmax_global
            ax.imshow(d, cmap="seismic", vmin=-vmax, vmax=+vmax)
            _draw_scale_bar(ax, d.shape, pixel_size_um, bar_um)
        axes_methods.append((ax, lbl))

    # RAW
    ax_raw = fig.add_axes(rect_raw)
    ax_raw.axis("off")
    d = diffs.get("raw")
    if d is None:
        ax_raw.text(0.5, 0.5, "missing", ha="center", va="center",
                    color="red", fontsize=12, transform=ax_raw.transAxes)
    else:
        if vmax_fixed is not None:
            vmax = float(vmax_fixed) if vmax_fixed > 0 else 1e-6
        elif scale == "perpanel":
            vmax = float(np.percentile(np.abs(d), 99)) or 1e-6
        else:
            vmax = vmax_global
        ax_raw.imshow(d, cmap="seismic", vmin=-vmax, vmax=+vmax)
        _draw_scale_bar(ax_raw, d.shape, pixel_size_um, bar_um)

    # Labels
    for ax, lbl in axes_methods:
        bbox = ax.get_position()
        x = (bbox.x0 + bbox.x1) / 2
        y = bbox.y1 + (label_pad_top * 0.6)
        fig.text(x, y, lbl, ha="center", va="bottom", fontsize=12, color="black")

    bbox = ax_raw.get_position()
    x = (bbox.x0 + bbox.x1) / 2
    y = bbox.y0 - (label_pad_bottom * 0.6)
    fig.text(x, y, "Raw", ha="center", va="top", fontsize=12, color="black")

    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


# -------------------------- CLI -------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Save two frames and their temporal diff per source from runs/")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--groups", nargs="+", required=True)

    raw = ap.add_mutually_exclusive_group(required=True)
    raw.add_argument("--raw-video", help="Path to raw video file for this video")
    raw.add_argument("--raw-template", help='Template like "data/raw/{category}/{video_id}.tif"')

    ap.add_argument("--select", default="latest", help="latest or best:<json.key>")
    ap.add_argument("--metric-mode", choices=["max","min"], default="max")
    ap.add_argument("--artifact-glob", default="artifacts/*.tif,artifacts/*.tiff,artifacts/*.mp4,artifacts/*.avi")
    ap.add_argument("--artifact-json-key", default=None)

    ap.add_argument("--frame-2", type=int, default=399, help="Use this as the second frame index (N)")
    ap.add_argument("--outdir", default="results_frames")

    # Scale bar
    ap.add_argument("--pixel-size-um", type=float, required=True, help="Pixel size in microns (width=height)")
    ap.add_argument("--bar-um", type=float, default=None, help="Optional fixed scalebar length in microns; if omitted, auto-choose")

    # Output quality
    ap.add_argument("--dpi", type=int, default=600, help="PNG resolution (dots per inch)")
    ap.add_argument("--figsize-in", type=float, default=5.0, help="Per-image figure size in inches (square)")
    ap.add_argument("--save-svg", action="store_true", help="Also save vector SVG alongside PNGs")
    ap.add_argument("--save-pdf", action="store_true", help="Also save vector PDF alongside PNGs")

    # Summary options
    ap.add_argument("--no-summary", action="store_true", help="Disable saving the method+raw frame summary SVG")
    ap.add_argument("--summary-frame", choices=["0", "N"], default="N", help="Which frame to show in summary (0 or N)")
    ap.add_argument("--summary-svg-name", default=None, help="Filename for the frame summary SVG (default auto)")
    ap.add_argument("--summary-figsize-in", type=float, default=7.0, help="Summary figure size in inches (square)")

    # Diff summary options
    ap.add_argument("--no-diff-summary", action="store_true", help="Disable saving the signed diff summary SVG")
    ap.add_argument("--diff-summary-scale", choices=["perpanel", "global"], default="perpanel",
                    help="Color scaling for signed diff summary")
    ap.add_argument("--diff-summary-mode", choices=["temporal", "vsraw"], default="temporal",
                    help="Use temporal (frame0−frameN) or per-frame method-vs-raw differences")
    ap.add_argument("--diff-summary-vmax", type=float, default=None,
                    help="Override vmax for signed diffs (data units). If set, used for all panels.")
    return ap.parse_args()


# -------------------------- main -------------------------- #

def main():
    args = parse_args()

    base_out = os.path.join(args.outdir, args.experiment, args.category, args.video_id)
    ensure_dir(base_out)

    # Load RAW frames
    raw_path = args.raw_video or args.raw_template.format(category=args.category, video_id=args.video_id)
    raw_frames = load_two_frames(raw_path, frame2=args.frame_2)
    Ht, Wt = raw_frames[0].shape

    # Save RAW
    raw_out = os.path.join(base_out, "raw")
    ensure_dir(raw_out)
    save_npy_png_bar(raw_frames[0], os.path.join(raw_out, "frame_000.npy"), os.path.join(raw_out, "frame_000.png"),
                     pixel_size_um=args.pixel_size_um, bar_um=args.bar_um,
                     dpi=args.dpi, figsize_in=args.figsize_in, save_svg=args.save_svg, save_pdf=args.save_pdf)
    save_npy_png_bar(raw_frames[1], os.path.join(raw_out, f"frame_{args.frame_2:03d}.npy"),
                     os.path.join(raw_out, f"frame_{args.frame_2:03d}.png"),
                     pixel_size_um=args.pixel_size_um, bar_um=args.bar_um,
                     dpi=args.dpi, figsize_in=args.figsize_in, save_svg=args.save_svg, save_pdf=args.save_pdf)
    save_diff_images_bar(raw_frames[0], raw_frames[1], raw_out, prefix="raw",
                         pixel_size_um=args.pixel_size_um, bar_um=args.bar_um,
                         dpi=args.dpi, figsize_in=args.figsize_in, save_svg=args.save_svg, save_pdf=args.save_pdf)

    # Discover runs and select one per group
    selected: dict[str, str] = {}
    for g in args.groups:
        run_dirs = discover_runs(args.runs_root, g, args.experiment, args.category, args.video_id)
        pick = pick_run(run_dirs, args.select, args.metric_mode)
        if pick is None:
            print(f"[WARN] No runs for group {g}")
            continue
        selected[g] = pick
        print(f"{g}: using {os.path.basename(pick)}")

    # For each method, load frames, resize to RAW for visualization, and save
    methods_aligned: dict[str, list[np.ndarray]] = {}
    for g, rundir in selected.items():
        art = find_artifact(rundir, args.artifact_glob, args.artifact_json_key)
        if art is None:
            print(f"[WARN] No artifact video for {g} in {rundir}")
            continue

        frs = load_two_frames(art, frame2=args.frame_2)
        frs = [cv2.resize(f, (Wt, Ht), interpolation=cv2.INTER_AREA) if f.shape != (Ht, Wt) else f for f in frs]
        methods_aligned[g.lower()] = frs  # 'ants', 'cotracker', 'normcorre', 'lddmms'

        out_dir = os.path.join(base_out, g)
        ensure_dir(out_dir)
        save_npy_png_bar(frs[0], os.path.join(out_dir, "frame_000.npy"), os.path.join(out_dir, "frame_000.png"),
                         pixel_size_um=args.pixel_size_um, bar_um=args.bar_um,
                         dpi=args.dpi, figsize_in=args.figsize_in, save_svg=args.save_svg, save_pdf=args.save_pdf)
        save_npy_png_bar(frs[1], os.path.join(out_dir, f"frame_{args.frame_2:03d}.npy"),
                         os.path.join(out_dir, f"frame_{args.frame_2:03d}.png"),
                         pixel_size_um=args.pixel_size_um, bar_um=args.bar_um,
                         dpi=args.dpi, figsize_in=args.figsize_in, save_svg=args.save_svg, save_pdf=args.save_pdf)
        save_diff_images_bar(frs[0], frs[1], out_dir, prefix=g,
                             pixel_size_um=args.pixel_size_um, bar_um=args.bar_um,
                             dpi=args.dpi, figsize_in=args.figsize_in, save_svg=args.save_svg, save_pdf=args.save_pdf)
        print(f"Saved frames and diffs for {g} -> {out_dir}")

    # Build "2x2 methods + centered raw" summary of images
    if not args.no_summary:
        frame_idx = 0 if args.summary_frame == "0" else 1
        images_for_summary = {
            "ants":      methods_aligned.get("ants", [None, None])[frame_idx] if "ants" in methods_aligned else None,
            "cotracker": methods_aligned.get("cotracker", [None, None])[frame_idx] if "cotracker" in methods_aligned else None,
            "lddmms":    methods_aligned.get("lddmms", [None, None])[frame_idx] if "lddmms" in methods_aligned else None,
            "normcorre": methods_aligned.get("normcorre", [None, None])[frame_idx] if "normcorre" in methods_aligned else None,
            "raw":       raw_frames[frame_idx],
        }
        svg_name = args.summary_svg_name or f"summary_frame{'0' if frame_idx == 0 else 'N'}.svg"
        out_svg = os.path.join(base_out, svg_name)
        save_summary_grid_2x2plusraw_center(
            images_for_summary, out_svg,
            pixel_size_um=args.pixel_size_um, bar_um=args.bar_um,
            figsize_in=args.summary_figsize_in
        )
        print("Saved summary SVG:", out_svg)

    # Signed differences summary (2x2 methods + centered raw)
    if not args.no_diff_summary:
        frames0 = {
            "ants":      methods_aligned.get("ants", [None, None])[0] if "ants" in methods_aligned else None,
            "cotracker": methods_aligned.get("cotracker", [None, None])[0] if "cotracker" in methods_aligned else None,
            "lddmms":    methods_aligned.get("lddmms", [None, None])[0] if "lddmms" in methods_aligned else None,
            "normcorre": methods_aligned.get("normcorre", [None, None])[0] if "normcorre" in methods_aligned else None,
            "raw":       raw_frames[0],
        }
        framesN = {
            "ants":      methods_aligned.get("ants", [None, None])[1] if "ants" in methods_aligned else None,
            "cotracker": methods_aligned.get("cotracker", [None, None])[1] if "cotracker" in methods_aligned else None,
            "lddmms":    methods_aligned.get("lddmms", [None, None])[1] if "lddmms" in methods_aligned else None,
            "normcorre": methods_aligned.get("normcorre", [None, None])[1] if "normcorre" in methods_aligned else None,
            "raw":       raw_frames[1],
        }
        out_svg_diff = os.path.join(base_out, f"summary_diffs_signed_{args.diff_summary_mode}_{args.diff_summary_scale}.svg")
        save_summary_grid_signed_diffs_2x2plusraw_center(
            frames0, framesN, raw_frames[0], raw_frames[1], out_svg_diff,
            pixel_size_um=args.pixel_size_um, bar_um=args.bar_um,
            figsize_in=args.summary_figsize_in,
            scale=args.diff_summary_scale, mode=args.diff_summary_mode,
            vmax_fixed=args.diff_summary_vmax,
        )
        print("Saved summary signed-diff SVG:", out_svg_diff)

    print("Done. Check:", base_out)


if __name__ == "__main__":
    main()
