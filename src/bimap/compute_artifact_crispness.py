#!/usr/bin/env python3
"""
compute_artifact_crispness_normcorre.py

Traverse only:
  output/normcorre/<experiment>/<category>/<video_id>/run_<id>/

For each run, process every artifact video in artifacts/ except a file named exactly 'video.mp4'.

Computes (per frame):
  - crispness_input (absolute)
  - crispness_artifact (absolute)
  - Δ crispness (%) vs input
  - correlation with mean artifact frame (Pearson)

Saves:
  * per-artifact CSV: per_frame_crispness_<artifact>.csv
  * per-artifact JSON: artifact_metrics_<artifact>.json
  * per-run CSV:       artifact_crispness_index.csv
  * global CSV:        output/normcorre/artifact_crispness_index.csv

Notes:
  - Uses from utils import load_video, crispness
  - Robustly resolves config.json["data"]["path"] relative to the project root (parent of 'output/').
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from utils import load_video, crispness  # <- your loader + metric

VID_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".tif", ".tiff")
SKIP_NAME = "video.mp4"

# ---------------- basic IO ---------------- #

def read_json(path: Path) -> dict:
    if not path.exists(): return {}
    try: return json.loads(path.read_text())
    except Exception: return {}

def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))

# ---------------- helpers: arrays ---------------- #

def as_float64(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64, copy=False) if x.dtype != np.float64 else x

def squeeze_keep_3d4d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 4 or (a.ndim > 2 and 1 in a.shape):
        axes = [i for i, s in enumerate(a.shape) if s == 1]
        if not axes: break
        if a.ndim <= 3: break
        a = np.squeeze(a, axis=axes[0])
    return a

def normalize_video_shape(arr: np.ndarray) -> np.ndarray:
    """
    Normalize to (T,H,W) or (T,H,W,C).
    Accepts common layouts: (1,T,H,W), (H,W,T), (T,C,H,W), (C,T,H,W), (H,W,T,C), (C,H,W,T), etc.
    """
    a = squeeze_keep_3d4d(np.asarray(arr))
    if a.ndim == 3:
        sh = a.shape
        t_axis = int(np.argmax(sh))
        if t_axis == 0:   # (T,H,W)
            return a
        if t_axis == 2:   # (H,W,T) -> (T,H,W)
            return np.moveaxis(a, 2, 0)
        # (H,T,W) -> (T,H,W)
        return np.moveaxis(a, 1, 0)
    if a.ndim == 4:
        sh = list(a.shape)
        # already (T,H,W,C) with small C last
        if sh[-1] <= 4 and sh[0] >= 2:
            return a
        chan_axes = [i for i, s in enumerate(sh) if s <= 4]
        t_axis = int(np.argmax(sh))
        if t_axis == 0 and 1 in chan_axes:       # (T,C,H,W) -> (T,H,W,C)
            return np.moveaxis(a, 1, -1)
        if t_axis == 1 and 0 in chan_axes:       # (C,T,H,W) -> (T,H,W,C)
            a = np.moveaxis(a, 1, 0)
            a = np.moveaxis(a, 0, -1)
            return a
        if t_axis == 3:                          # (...,T) -> T first
            return np.moveaxis(a, 3, 0)
        if t_axis == 2 and sh[-1] <= 4:          # (H,W,T,C) -> (T,H,W,C)
            return np.moveaxis(a, 2, 0)
        # fallback: time first, small-dim to last
        a = np.moveaxis(a, t_axis, 0)
        sh2 = list(a.shape)
        cand = [i for i, s in enumerate(sh2) if i != 0 and s <= 4]
        if cand:
            a = np.moveaxis(a, cand[0], -1)
        return a
    return a

def channels_equal(img: np.ndarray, atol: float = 1e-8) -> bool:
    if img.ndim != 3 or img.shape[-1] < 3:
        return False
    c0 = img[..., 0]
    return np.allclose(c0, img[..., 1], atol=atol, rtol=0.0) and np.allclose(c0, img[..., 2], atol=atol, rtol=0.0)

def collapse_rgb_grayscale(vol: np.ndarray) -> np.ndarray:
    """
    If the video is grayscale but saved as RGB (all channels equal), collapse to single channel.
    Checks a few frames for speed.
    """
    if vol.ndim == 4 and vol.shape[-1] >= 3:
        T = vol.shape[0]
        idxs = [0, T // 2, T - 1] if T >= 3 else list(range(T))
        for i in idxs:
            if not channels_equal(vol[i]):
                return vol
        return vol[..., 0]
    return vol

def ensure_channel_compat(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Match channel counts between two frames; tile single-channel to multi if needed."""
    if a.ndim == 2 and b.ndim == 2:
        return a, b
    if a.ndim == 3 and b.ndim == 3:
        ca, cb = a.shape[-1], b.shape[-1]
        if ca == cb: return a, b
        if ca == 1 and cb > 1: return np.repeat(a, cb, axis=-1), b
        if cb == 1 and ca > 1: return a, np.repeat(b, ca, axis=-1)
        c = min(ca, cb); return a[..., :c], b[..., :c]
    if a.ndim == 2 and b.ndim == 3:
        return np.repeat(a[..., None], b.shape[-1], axis=-1), b
    if b.ndim == 2 and a.ndim == 3:
        return a, np.repeat(b[..., None], a.shape[-1], axis=-1)
    return np.squeeze(a), np.squeeze(b)

def center_crop_to_match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center-crop two frames to the same (H,W)."""
    def hw(x):
        x = np.squeeze(x)
        return (x.shape[0], x.shape[1])
    Ha, Wa = hw(a); Hb, Wb = hw(b)
    H = min(Ha, Hb); W = min(Wa, Wb)
    def crop(x):
        x = np.squeeze(x)
        hh, ww = x.shape[0], x.shape[1]
        top = (hh - H) // 2; left = (ww - W) // 2
        return x[top:top+H, left:left+W] if x.ndim == 2 else x[top:top+H, left:left+W, ...]
    return crop(a), crop(b)

def crop_video_to_common(vol: np.ndarray) -> np.ndarray:
    """Ensure uniform HxW across frames by center-cropping to min H,W if needed."""
    T = vol.shape[0]
    if vol.ndim == 4:
        Hs = [vol[t].shape[0] for t in range(T)]
        Ws = [vol[t].shape[1] for t in range(T)]
        H, W = min(Hs), min(Ws)
        if all(h == H and w == W for h, w in zip(Hs, Ws)): return vol
        out = np.empty((T, H, W, vol.shape[-1]), dtype=vol.dtype)
        for t in range(T):
            f = vol[t]; hh, ww = f.shape[0], f.shape[1]
            top = (hh - H) // 2; left = (ww - W) // 2
            out[t] = f[top:top+H, left:left+W, :]
        return out
    elif vol.ndim == 3:
        Hs = [vol[t].shape[0] for t in range(T)]
        Ws = [vol[t].shape[1] for t in range(T)]
        H, W = min(Hs), min(Ws)
        if all(h == H and w == W for h, w in zip(Hs, Ws)): return vol
        out = np.empty((T, H, W), dtype=vol.dtype)
        for t in range(T):
            f = vol[t]; hh, ww = f.shape[0], f.shape[1]
            top = (hh - H) // 2; left = (ww - W) // 2
            out[t] = f[top:top+H, left:left+W]
        return out
    return vol

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two same-shaped images (any channels)."""
    x = a.reshape(-1); y = b.reshape(-1)
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(xm) * np.linalg.norm(ym)
    return float(np.dot(xm, ym) / denom) if denom > 1e-12 else float("nan")

def to_crispness_image(img: np.ndarray) -> np.ndarray:
    """Make sure crispness (Fro norm) sees a 2-D image: average channels if present."""
    x = np.squeeze(img)
    if x.ndim == 3:  # H,W,C
        x = x.mean(axis=-1)
    return x

# ---------------- path resolution ---------------- #

def _sanitize_spec(spec: str) -> str:
    """Strip outer quotes, normalize slashes, collapse //."""
    s = str(spec).strip()
    if (len(s) >= 2) and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    s = s.replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    return s

def resolve_data_path(run_dir: Path, data_path: str, verbose: bool = False) -> Optional[Path]:
    """
    Resolve config['data']['path'] relative to:
      - run_dir
      - output_root (run_dir.parents[4])
      - project_root (parent of output_root)
      - project_root/'data'
      - project_root/'data/input'
      - project_root/'input'
    Also re-root prefixes containing 'data/input/' or 'input/' under project_root.
    """
    if not data_path:
        return None

    spec = _sanitize_spec(data_path)
    p = Path(spec)
    if p.is_absolute():
        return p if p.exists() else None

    # Identify roots correctly:
    # run_dir = output/<group>/<exp>/<cat>/<vid>/run_<id>
    # parents[4] -> output root; parents[5] -> project root
    try:
        output_root = run_dir.parents[4]
        project_root = run_dir.parents[5]
    except Exception:
        output_root = run_dir
        project_root = run_dir.parent

    tried: List[Path] = []

    def try_path(base: Path, extra: str | Path) -> Optional[Path]:
        cand = (base / extra).resolve()
        tried.append(cand)
        return cand if cand.exists() else None

    # Direct attempts in plausible bases
    for base in (run_dir, output_root, project_root, project_root / "data", project_root / "data/input", project_root / "input"):
        found = try_path(base, spec)
        if found:
            if verbose: print(f"[resolve] OK: {found}")
            return found

    # Re-root for prefixes like ../../data/input/... or data/input/... or input/...
    norm = spec
    # extract tail after the *last* occurrence of 'data/input/' or 'input/'
    tail = None
    if "data/input/" in norm:
        tail = norm.split("data/input/")[-1]
        anchors = [project_root / "data/input"]
    elif "/input/" in norm:
        tail = norm.split("/input/")[-1]
        anchors = [project_root / "input", project_root / "data/input"]
    else:
        anchors = []

    if tail is not None:
        for anchor in anchors:
            found = try_path(anchor, tail)
            if found:
                if verbose: print(f"[resolve] OK (re-rooted): {found}")
                return found

    if verbose:
        print("[resolve] FAILED. Tried:")
        for c in tried:
            print("   -", c)
    return None

# ---------------- per-artifact processing ---------------- #

def process_one_artifact(run_dir: Path, art_path: Path, verbose: bool = False) -> Optional[Dict]:
    # Load artifact
    try:
        art_vid, _, _ = load_video(str(art_path))
    except Exception:
        if verbose: print(f"[load] failed artifact: {art_path}")
        return None

    # Load input reference via config
    cfg = read_json(run_dir / "config.json")
    ref_rel = cfg.get("data", {}).get("path") if isinstance(cfg.get("data"), dict) else None
    ref_path = resolve_data_path(run_dir, ref_rel, verbose=verbose) if ref_rel else None
    if not ref_path or not ref_path.exists():
        if verbose: print(f"[resolve] could not resolve ref path from '{ref_rel}' at {run_dir}")
        return None

    try:
        ref_vid, _, _ = load_video(str(ref_path))
    except Exception:
        if verbose: print(f"[load] failed input: {ref_path}")
        return None

    # Normalize/collapse
    art_vid = collapse_rgb_grayscale(normalize_video_shape(as_float64(art_vid)))
    ref_vid = collapse_rgb_grayscale(normalize_video_shape(as_float64(ref_vid)))

    # Time-align
    T = min(art_vid.shape[0], ref_vid.shape[0])
    if T == 0:
        return None
    art_vid = art_vid[:T]
    ref_vid = ref_vid[:T]

    # Uniform HxW for mean computation
    art_vid = crop_video_to_common(art_vid)
    mean_art = np.mean(art_vid, axis=0)

    crisp_in_list, crisp_out_list, crisp_delta_list, corr_list = [], [], [], []
    for t in range(T):
        in_f, out_f = ref_vid[t], art_vid[t]
        # crop & channel align for fair comparison
        in_f, out_f = center_crop_to_match(in_f, out_f)
        in_f, out_f = ensure_channel_compat(in_f, out_f)

        # absolute crispness on 2D (your utils.crispness expects 2D)
        c_in = float(crispness(to_crispness_image(in_f)))
        c_out = float(crispness(to_crispness_image(out_f)))
        delta_pct = (c_out - c_in) / c_in * 100.0 if c_in > 1e-12 else float("nan")

        # correlation with mean (crop/align mean to out_f)
        mo, out_cm = center_crop_to_match(mean_art, out_f)
        mo, out_cm = ensure_channel_compat(mo, out_cm)
        r = pearson_corr(out_cm, mo)

        crisp_in_list.append(c_in)
        crisp_out_list.append(c_out)
        crisp_delta_list.append(delta_pct)
        corr_list.append(r)

    # Save per-frame CSV
    safe = art_path.stem.replace(" ", "_")
    per_frame = pd.DataFrame({
        "frame_idx": np.arange(T, dtype=int),
        "crispness_input": crisp_in_list,
        "crispness_artifact": crisp_out_list,
        "crispness_delta_percent": crisp_delta_list,
        "corr_with_mean": corr_list,
    })
    per_frame.to_csv(run_dir / f"per_frame_crispness_{safe}.csv", index=False)

    # Summaries
    def summarize(vals: List[float]) -> Tuple[float, float]:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return (float(arr.mean()), float(arr.std(ddof=0))) if arr.size else (float("nan"), float("nan"))

    ci_m, ci_s = summarize(crisp_in_list)
    co_m, co_s = summarize(crisp_out_list)
    dp_m, dp_s = summarize(crisp_delta_list)
    cr_m, cr_s = summarize(corr_list)

    summary = {
        "artifact_name": art_path.name,
        "frames": int(T),
        "crispness_input_mean": ci_m,
        "crispness_input_std": ci_s,
        "crispness_artifact_mean": co_m,
        "crispness_artifact_std": co_s,
        "crispness_delta_percent_mean": dp_m,
        "crispness_delta_percent_std": dp_s,
        "corr_with_mean_mean": cr_m,
        "corr_with_mean_std": cr_s,
    }

    write_json(run_dir / f"artifact_metrics_{safe}.json", {"ok": True, "metrics": summary})
    return summary

# ---------------- traversal ---------------- #

def find_normcorre_runs(root: Path) -> List[Path]:
    runs: List[Path] = []
    norm_root = root / "normcorre"
    if not norm_root.exists(): return runs
    for exp in sorted([p for p in norm_root.iterdir() if p.is_dir()]):
        for cat in sorted([p for p in exp.iterdir() if p.is_dir()]):
            for vid in sorted([p for p in cat.iterdir() if p.is_dir()]):
                for run in sorted(vid.glob("run_*")):
                    if run.is_dir(): runs.append(run)
    return runs

def list_artifacts(run_dir: Path) -> List[Path]:
    arts = run_dir / "artifacts"
    if not arts.exists(): return []
    vids = [p for p in arts.iterdir() if p.suffix.lower() in VID_EXTS]
    return sorted([p for p in vids if p.name != SKIP_NAME])

# ---------------- main ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output", help="Experiments root (default: output)")
    ap.add_argument("--verbose", action="store_true", help="Print resolution attempts")
    ap.add_argument("--limit_runs", type=int, default=0, help="Process at most N runs (0 = all)")
    args = ap.parse_args()

    root = Path(args.root)
    runs = find_normcorre_runs(root)
    if args.limit_runs > 0:
        runs = runs[:args.limit_runs]

    global_rows = []
    for run_dir in runs:
        artifacts = list_artifacts(run_dir)
        if not artifacts:
            print(f"[skip] {run_dir} (no artifact videos except {SKIP_NAME})")
            continue

        per_run_rows = []
        for art in artifacts:
            res = process_one_artifact(run_dir, art, verbose=args.verbose)
            if not res:
                print(f"[skip] {run_dir} :: {art.name}")
                continue
            row = {
                "experiment": run_dir.parents[3].name,
                "category": run_dir.parents[2].name,
                "video_id": run_dir.parents[1].name,
                "run_id": run_dir.name,
                **res
            }
            per_run_rows.append(row)
            global_rows.append({**row, "run_dir": str(run_dir)})
            print(f"[ok] {row['experiment']}/{row['category']}/{row['video_id']}/{row['run_id']} :: {res['artifact_name']}")

        if per_run_rows:
            df_run = pd.DataFrame(per_run_rows)
            (run_dir / "artifact_crispness_index.csv").write_text(df_run.to_csv(index=False))

    if global_rows:
        df = pd.DataFrame(global_rows)
        out_csv = root / "normcorre" / "artifact_crispness_index.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved {len(df)} artifact summaries to {out_csv}")
    else:
        print("No artifacts processed.")

if __name__ == "__main__":
    main()
