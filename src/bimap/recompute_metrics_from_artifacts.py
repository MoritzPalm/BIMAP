#!/usr/bin/env python3
"""
Recompute per-frame metrics (SSIM, MSE, absolute crispness for ref & output,
crispness % change vs uncorrected, correlation-with-mean) for each run under:
  output/<group>/<experiment>/<category>/<video_id>/run_<id>/

Artifact assumption:
  - artifacts/ contains a single video or multi-page TIFF (no image-sequence folders).
  - We use cotracker.utils.visualizer.read_video_from_path for ALL videos and .tif/.tiff.

Reference frames:
  - Loaded ONLY from config.json["data"]["path"] and resolved robustly:
      1) relative to the run directory,
      2) relative to the output/ directory,
      3) relative to the project root (sibling of output/).
    We also re-root patterns like "../../data/input/..." or "../../input/...".
  - Never uses config["run"].

Outputs per run:
  - per_frame_recomputed.csv with columns:
      frame_idx, ssim, mse,
      crispness_ref, crispness_out, crispness_delta_percent,
      corr_with_mean
  - recomputed_result.json (means/stds)

Global index:
  - output/recompute_index.csv

Important handling:
  - Shapes are normalized to (T,H,W) or (T,H,W,C).
  - If a video is grayscale-but-saved-as-RGB (all 3 channels equal),
    we collapse to single channel automatically (within tolerance).
  - SSIM uses channel_axis for color; single-channel uses 2D SSIM.
  - Intensities are not rescaled, but we cast to float64; SSIM data_range
    is derived per-pair from the output frame to avoid degenerate ranges.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import mean_squared_error as mse_metric

from cotracker.utils.visualizer import read_video_from_path
from utils import crispness  # your crispness(image) implementation

# ------------------------- IO helpers -------------------------

VID_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".tif", ".tiff")

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))

def list_videos_in(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in VID_EXTS])

# ------------------------- dtype/shape utilities -------------------------

def as_float64(arr: np.ndarray) -> np.ndarray:
    """Convert to float64 without changing scale."""
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64, copy=False)
    return arr

def squeeze_singletons_keep_3d4d(arr: np.ndarray) -> np.ndarray:
    """
    Squeeze singleton axes but keep 3D/4D forms (avoid collapsing T/H/W/C to <3 dims).
    """
    arr = np.asarray(arr)
    while arr.ndim > 4 or (arr.ndim > 2 and 1 in arr.shape):
        # remove the first singleton axis we find (except keep at least 3 dims)
        axes = [i for i, s in enumerate(arr.shape) if s == 1]
        if not axes:
            break
        if arr.ndim <= 3:
            # if removing would drop to 2D, stop
            break
        arr = np.squeeze(arr, axis=axes[0])
    return arr

def normalize_video_shape(arr: np.ndarray) -> np.ndarray:
    """
    Normalize common video layouts to (T,H,W) or (T,H,W,C).

    Accepts:
      (T,H,W,C), (T,H,W),
      (1,T,H,W) -> (T,H,W),
      (H,W,T)   -> (T,H,W),
      (H,W,C,T) -> (T,H,W,C),
      (T,C,H,W) -> (T,H,W,C),
      (C,T,H,W) -> (T,H,W,C),
      (C,H,W,T) -> (T,H,W,C)
    """
    arr = squeeze_singletons_keep_3d4d(np.asarray(arr))

    if arr.ndim == 3:
        # Could be (T,H,W) or (H,W,T)
        sh = arr.shape
        t_axis = int(np.argmax(sh))  # largest dimension is time
        if t_axis == 0:
            return arr
        elif t_axis == 2:
            return np.moveaxis(arr, 2, 0)  # (H,W,T)->(T,H,W)
        else:
            return np.moveaxis(arr, 1, 0)  # (H,T,W)->(T,H,W)

    if arr.ndim == 4:
        sh = list(arr.shape)
        # Fast path: already (T,H,W,C) with small C last
        if sh[-1] <= 4 and sh[0] >= 2:
            return arr

        # Identify likely channel axes (size <= 4)
        chan_axes = [i for i, s in enumerate(sh) if s <= 4]
        # Time = largest axis
        t_axis = int(np.argmax(sh))

        # (T,C,H,W) -> move C (axis 1) to last
        if t_axis == 0 and 1 in chan_axes:
            return np.moveaxis(arr, 1, -1)

        # (C,T,H,W) -> move T (axis 1) to 0, C (axis 0) to -1
        if t_axis == 1 and 0 in chan_axes:
            arr = np.moveaxis(arr, 1, 0)  # T first
            arr = np.moveaxis(arr, 0, -1) # C last
            return arr

        # (H,W,C,T) -> move T (axis 3) to 0
        if t_axis == 3:
            return np.moveaxis(arr, 3, 0)

        # (C,H,W,T) -> move T (axis 3) to 0, C (axis 0) to -1
        if t_axis == 3 and 0 in chan_axes:
            arr = np.moveaxis(arr, 3, 0)
            arr = np.moveaxis(arr, 0, -1)
            return arr

        # (H,W,T,C) -> move T (axis 2) to 0, keep C last
        if t_axis == 2 and sh[-1] <= 4:
            return np.moveaxis(arr, 2, 0)

        # Fallback: put time (largest) first, put a small-dim (<=4) last if exists
        arr = np.moveaxis(arr, t_axis, 0)
        sh2 = list(arr.shape)
        cand = [i for i, s in enumerate(sh2) if i != 0 and s <= 4]
        if cand:
            arr = np.moveaxis(arr, cand[0], -1)
        return arr

    return arr  # unexpected; metrics handle squeezing later

def channels_equal(img: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Return True if img is (H,W,3/4) and all first 3 channels are equal within tolerance.
    """
    if img.ndim != 3 or img.shape[-1] < 3:
        return False
    c0 = img[..., 0]
    for c in range(1, 3):
        if not np.allclose(c0, img[..., c], atol=atol, rtol=0.0):
            return False
    return True

def collapse_rgb_grayscale(vol: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    """
    If a video is grayscale-but-saved-as-RGB (all channels equal), drop to single-channel.
    Works for (T,H,W,3/4). Keeps (T,H,W) unchanged.
    """
    if vol.ndim == 4 and vol.shape[-1] >= 3:
        # check on a few frames for speed
        T = vol.shape[0]
        idxs = [0, T//2, T-1] if T >= 3 else list(range(T))
        is_gray_rgb = True
        for i in idxs:
            if not channels_equal(vol[i], atol=atol):
                is_gray_rgb = False
                break
        if is_gray_rgb:
            return vol[..., 0]  # (T,H,W)
    return vol

def ensure_channel_compat(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a and b have the same number of channels (if any).
    If one is (H,W) and the other (H,W,C), tile the single-channel to C or crop to min C.
    """
    if a.ndim == 2 and b.ndim == 2:
        return a, b
    if a.ndim == 3 and b.ndim == 3:
        ca = a.shape[-1]; cb = b.shape[-1]
        if ca == cb:
            return a, b
        if ca == 1 and cb > 1:
            a = np.repeat(a, cb, axis=-1); return a, b
        if cb == 1 and ca > 1:
            b = np.repeat(b, ca, axis=-1); return a, b
        c = min(ca, cb)
        return a[..., :c], b[..., :c]
    if a.ndim == 2 and b.ndim == 3:
        c = b.shape[-1]
        a = np.repeat(a[..., None], c, axis=-1)
        return a, b
    if b.ndim == 2 and a.ndim == 3:
        c = a.shape[-1]
        b = np.repeat(b[..., None], c, axis=-1)
        return a, b
    return np.squeeze(a), np.squeeze(b)

def center_crop_spatial(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center-crop a and b to the same HxW (min over spatial dims)."""
    def get_hw(x):
        if x.ndim == 2:
            return x.shape[0], x.shape[1]
        elif x.ndim == 3:
            return x.shape[0], x.shape[1]  # assuming (H,W,C)
        else:
            x = np.squeeze(x)
            return x.shape[0], x.shape[1]

    Ha, Wa = get_hw(a)
    Hb, Wb = get_hw(b)
    H = int(min(Ha, Hb)); W = int(min(Wa, Wb))

    def crop(x):
        if x.ndim == 2:
            hh, ww = x.shape
            top = (hh - H) // 2; left = (ww - W) // 2
            return x[top:top+H, left:left+W]
        elif x.ndim == 3:
            hh, ww = x.shape[0], x.shape[1]
            top = (hh - H) // 2; left = (ww - W) // 2
            return x[top:top+H, left:left+W, ...]
        else:
            y = np.squeeze(x)
            hh, ww = y.shape[0], y.shape[1]
            top = (hh - H) // 2; left = (ww - W) // 2
            return y[top:top+H, left:left+W]

    return crop(a), crop(b)

def crop_all_to_common_video(vol: np.ndarray) -> np.ndarray:
    """
    Ensure a consistent spatial size across frames by center-cropping to min HxW if needed.
    """
    T = vol.shape[0]
    if vol.ndim == 4:
        Hs = [vol[t].shape[0] for t in range(T)]
        Ws = [vol[t].shape[1] for t in range(T)]
        H = min(Hs); W = min(Ws)
        if all(h == H and w == W for h, w in zip(Hs, Ws)):
            return vol
        out = np.empty((T, H, W, vol.shape[-1]), dtype=vol.dtype)
        for t in range(T):
            frame = vol[t]
            hh, ww = frame.shape[0], frame.shape[1]
            top = (hh - H) // 2; left = (ww - W) // 2
            out[t] = frame[top:top+H, left:left+W, :]
        return out
    elif vol.ndim == 3:
        Hs = [vol[t].shape[0] for t in range(T)]
        Ws = [vol[t].shape[1] for t in range(T)]
        H = min(Hs); W = min(Ws)
        if all(h == H and w == W for h, w in zip(Hs, Ws)):
            return vol
        out = np.empty((T, H, W), dtype=vol.dtype)
        for t in range(T):
            frame = vol[t]
            hh, ww = frame.shape[0], frame.shape[1]
            top = (hh - H) // 2; left = (ww - W) // 2
            out[t] = frame[top:top+H, left:left+W]
        return out
    else:
        return vol

# ------------------------- loading -------------------------

def load_video_or_stack(path: Path) -> np.ndarray:
    """Use your project's loader for ANY video or multi-page tiff."""
    return read_video_from_path(str(path))

def find_artifact_video(run_dir: Path) -> Optional[np.ndarray]:
    """Find a single video in artifacts/ and load it via read_video_from_path."""
    art = run_dir / "artifacts"
    vids = list_videos_in(art)
    if not vids:
        return None
    vpath = vids[0]
    try:
        vid = load_video_or_stack(vpath)
        return as_float64(vid)
    except Exception:
        return None

# ------------------------- data path resolution -------------------------

def resolve_data_path(run_dir: Path, data_path: str) -> Optional[Path]:
    """
    Resolve a possibly relative data path according to your layout:

    output/                           <- runs live here
    input/ or data/input/             <- reference stacks live here (sibling of output/)
    project_root/ (parent of output/) <- base for re-rooting relative paths
    """
    if not data_path:
        return None
    p = Path(data_path)
    if p.is_absolute():
        return p if p.exists() else None

    try:
        output_dir = run_dir.parents[5]         # output/
        project_root = output_dir.parent        # sibling of output/
    except Exception:
        output_dir = run_dir
        project_root = run_dir.parent

    # Try run_dir, output_dir, project_root
    for base in (run_dir, output_dir, project_root):
        cand = (base / data_path).resolve()
        if cand.exists():
            return cand

    # Re-root common prefixes
    norm = data_path.replace("\\", "/")
    prefixes = ("../../data/input/", "../data/input/", "data/input/",
                "../../input/", "../input/", "input/")
    for pref in prefixes:
        if norm.startswith(pref):
            remainder = norm[len(pref):]
            for anchor in ("data/input", "input"):
                cand = (project_root / anchor / remainder).resolve()
                if cand.exists():
                    return cand
            break
    return None

def load_reference_video(run_dir: Path) -> Optional[np.ndarray]:
    """
    Load reference (uncorrected) sequence from config.json["data"]["path"],
    using read_video_from_path for videos/tiffs. Never uses config["run"].
    """
    cfg = read_json(run_dir / "config.json")
    data_path = None
    if isinstance(cfg.get("data"), dict):
        data_path = cfg["data"].get("path")
    if not data_path:
        return None

    resolved = resolve_data_path(run_dir, data_path)
    if not resolved or not resolved.exists():
        return None

    try:
        vid = load_video_or_stack(resolved)
        return as_float64(vid)
    except Exception:
        return None

# ------------------------- metrics -------------------------

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation between two same-shaped images (any channels).
    Returns NaN if either input is (near-)constant.
    """
    x = a.reshape(-1)
    y = b.reshape(-1)
    x_m = x - x.mean()
    y_m = y - y.mean()
    x_norm = np.linalg.norm(x_m)
    y_norm = np.linalg.norm(y_m)
    if x_norm < 1e-12 or y_norm < 1e-12:
        return float("nan")
    return float(np.dot(x_m, y_m) / (x_norm * y_norm))

def compute_metrics_for_pair(ref: np.ndarray, out: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Return (ssim, mse, c_ref, c_out, crispness_delta_percent) for one frame pair.
    Assumes ref/out already center-cropped and channel-aligned.
    """
    # SSIM: use output dynamic range to avoid degenerate data_range
    dr = float(out.max() - out.min())
    if dr <= 0:
        dr = 1.0

    if ref.ndim == 3 and ref.shape[-1] > 1:
        ssim = ssim_metric(ref, out, data_range=dr, channel_axis=-1)
    else:
        ssim = ssim_metric(ref.squeeze(), out.squeeze(), data_range=dr)

    mse = mse_metric(ref, out)

    # Absolute crispness via your utils.crispness
    c_ref = float(crispness(ref))
    c_out = float(crispness(out))

    # Percent change vs uncorrected (reference)
    delta_c = float("nan")
    if c_ref > 1e-12:
        delta_c = (c_out - c_ref) / c_ref * 100.0

    return float(ssim), float(mse), c_ref, c_out, float(delta_c)

def summarize(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return float(arr.mean()), float(arr.std(ddof=0))

# ------------------------- per-run worker -------------------------

def process_run(run_dir: Path) -> Optional[Dict]:
    """Recompute metrics for a single run. Returns summary dict or None if skipped."""
    out_vid = find_artifact_video(run_dir)
    ref_vid = load_reference_video(run_dir)
    if out_vid is None or ref_vid is None:
        return None

    # Normalize shapes to (T,H,W) or (T,H,W,C)
    out_vid = normalize_video_shape(out_vid)
    ref_vid = normalize_video_shape(ref_vid)

    # If grayscale-saved-as-RGB (all channels equal), collapse to single-channel
    out_vid = collapse_rgb_grayscale(out_vid)
    ref_vid = collapse_rgb_grayscale(ref_vid)

    # Time-align: truncate to min length
    T = min(out_vid.shape[0], ref_vid.shape[0])
    out_vid = out_vid[:T]
    ref_vid = ref_vid[:T]

    # Ensure consistent spatial size in output (for mean computation)
    out_vid = crop_all_to_common_video(out_vid)
    mean_out = np.mean(out_vid, axis=0)

    ssim_list: List[float] = []
    mse_list: List[float] = []
    crisp_ref_list: List[float] = []
    crisp_out_list: List[float] = []
    crisp_delta_list: List[float] = []
    corr_mean_list: List[float] = []

    for t in range(T):
        rf = as_float64(ref_vid[t])
        of = as_float64(out_vid[t])

        # Center-crop to same spatial size & channel-align
        rf, of = center_crop_spatial(rf, of)
        rf, of = ensure_channel_compat(rf, of)

        # Metrics on the pair
        s, m, c_ref, c_out, c_pct = compute_metrics_for_pair(rf, of)
        ssim_list.append(s)
        mse_list.append(m)
        crisp_ref_list.append(c_ref)
        crisp_out_list.append(c_out)
        crisp_delta_list.append(c_pct)

        # Correlation-with-mean on output: crop mean_out to 'of' and align channels
        mo = mean_out
        mo, of_cm = center_crop_spatial(mo, of)
        mo, of_cm = ensure_channel_compat(mo, of_cm)
        corr = pearson_corr(of_cm, mo)
        corr_mean_list.append(corr)

    # Save per-frame CSV
    n = T
    per_frame = pd.DataFrame({
        "frame_idx": np.arange(n, dtype=int),
        "ssim": ssim_list,
        "mse": mse_list,
        "crispness_ref": crisp_ref_list,
        "crispness_out": crisp_out_list,
        "crispness_delta_percent": crisp_delta_list,
        "corr_with_mean": corr_mean_list,
    })
    per_frame.to_csv(run_dir / "per_frame_recomputed.csv", index=False)

    # Summaries
    ssim_mean, ssim_std = summarize(ssim_list)
    mse_mean, mse_std = summarize(mse_list)
    crisp_ref_mean, crisp_ref_std = summarize(crisp_ref_list)
    crisp_out_mean, crisp_out_std = summarize(crisp_out_list)
    crisp_delta_mean, crisp_delta_std = summarize(crisp_delta_list)
    corr_mean, corr_std = summarize(corr_mean_list)

    summary = {
        "frames": n,
        "m.ssim_mean": ssim_mean,
        "m.ssim_std": ssim_std,
        "m.mse_mean": mse_mean,
        "m.mse_std": mse_std,
        "m.crispness_ref_mean": crisp_ref_mean,
        "m.crispness_ref_std": crisp_ref_std,
        "m.crispness_out_mean": crisp_out_mean,
        "m.crispness_out_std": crisp_out_std,
        "m.crispness_delta_mean": crisp_delta_mean,   # percent change vs uncorrected
        "m.crispness_delta_std": crisp_delta_std,
        "m.corr_with_mean_mean": corr_mean,
        "m.corr_with_mean_std": corr_std,
    }

    # Write summary JSON (non-destructive)
    write_json(run_dir / "recomputed_result.json", {
        "ok": True,
        "from_artifacts": True,
        "metrics": {"summary": summary}
    })

    # For global index
    cfg = read_json(run_dir / "config.json")
    data_path_raw = cfg.get("data", {}).get("path") if isinstance(cfg.get("data"), dict) else None
    return {
        "group": run_dir.parents[4].name,        # output/<group>/<exp>/<cat>/<vid>/run_x
        "experiment": run_dir.parents[3].name,
        "category": run_dir.parents[2].name,
        "video_id": run_dir.parents[1].name,
        "run_id": run_dir.name.replace("run_", ""),
        "run_dir": str(run_dir),
        "data.path.raw": data_path_raw or "",
        **summary
    }

# ------------------------- traversal -------------------------

def find_run_dirs(root: Path) -> List[Path]:
    run_dirs: List[Path] = []
    if not root.exists():
        return []
    for group_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for exp_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()]):
            for cat_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
                for vid_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()]):
                    for run_dir in sorted(vid_dir.glob("run_*")):
                        if run_dir.is_dir():
                            run_dirs.append(run_dir)
    return run_dirs

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output", help="Root folder with runs (called 'output/').")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N runs (0 = all).")
    args = ap.parse_args()

    root = Path(args.root)
    run_dirs = find_run_dirs(root)
    if args.limit > 0:
        run_dirs = run_dirs[:args.limit]

    rows = []
    skipped = 0

    for run_dir in run_dirs:
        try:
            row = process_run(run_dir)
            if row is None:
                skipped += 1
                print(f"[skip] {run_dir} (no artifacts video or could not resolve config.json['data']['path'])")
                continue
            rows.append(row)
            print(f"[ok] {row['group']}/{row['experiment']}/{row['category']}/{row['video_id']}/{row['run_id']}")
        except Exception as e:
            skipped += 1
            print(f"[error:{type(e).__name__}] {run_dir}")

    # Write overall index
    if rows:
        df = pd.DataFrame(rows)
        out_csv = root / "recompute_index.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved {len(df)} summaries to {out_csv}")
    else:
        print("No runs processed successfully.")

    if skipped:
        print(f"Skipped runs: {skipped}")

if __name__ == "__main__":
    main()
