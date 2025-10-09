"""
Paper-ready tables with ONE ROW PER CONFIGURATION (read from config.json), averaged over experiments,
including metrics pulled correctly from result.json and per_frame.csv.

What it does
------------
1) Walks runs/ tree (no table.csv).
2) For each run, reads:
   - config.json  -> parameters (mapped into cfg.* columns)
   - result.json  -> metrics.summary{mse_mean, mse_std, crispness_improvement} + top-level runtime_s
   - per_frame.csv -> computes per-run SSIM and MSE (mean/std over frames); also fills input/output crispness if present
     (falls back to per_frame_recomputed.csv when necessary)
   - Loads the run's output artifact with utils.load_video and computes corr_with_mean itself (optional extra)
3) Aggregation:
   - First average within each experiment (exp_name) -> one row per (group, module, exp_name, configuration).
   - Then compute mean ± std ACROSS EXPERIMENTS for each (group, module, configuration).
4) Builds per-method tables with one row per configuration (cfg.*), reporting:
   - SSIM, MSE, Crispness (means, stds, and "mean ± std") across experiments,
   - Runtime (mean ± std) across experiments,
   - crispness_improvement from result.json (mean ± std) across experiments,
   - corr_with_mean (mean of means/stds across experiments, if available).
5) Writes one CSV per method and a combined CSV.

Outputs
-------
- runs/paper_<group>_<module>.csv
- runs/paper_all_methods.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Import utilities for computing input crispness from the raw video
from utils import crispness, load_video

# ----------------- config -----------------
CRISPNESS_CANON_COL = "m.crispness"          # processed/output crispness (per-run; later aggregated)
CRISPNESS_INPUT_COL = "m.crispness_input"    # uncorrected input crispness (per-run; later aggregated)
RUNTIME = "runtime_s"

REQUIRED_PARAM_COLS = [
    "cfg.template_strategy",
    "cfg.gaussian_filtered",
    "cfg.diff_warp",
    "cfg.visibility",
    "cfg.grid_size",
    "cfg.method",
]

MAX_PARAM_UNIQUE = 10
MAX_NUMERIC_UNIQUE = 8
MAX_TOTAL_PARAM_COLS = 12  # INCLUDING required params


# ----------------- helpers -----------------
def flatten_dict(d: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep))
        else:
            out[key] = v
    return out


def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def load_result_json(run_dir: Path) -> dict:
    return load_json(run_dir / "result.json")


def load_config_json(run_dir: Path) -> dict:
    return load_json(run_dir / "config.json")


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def _align_fill(main: pd.DataFrame, fb: pd.DataFrame) -> pd.DataFrame:
    if main.empty and fb.empty:
        return pd.DataFrame({"frame_idx": []})
    if main.empty:
        return fb
    if fb.empty:
        if "frame_idx" not in main.columns:
            main = main.copy()
            main["frame_idx"] = np.arange(len(main))
        return main

    if "frame_idx" in main.columns and "frame_idx" in fb.columns:
        merged = pd.merge(fb, main, on="frame_idx", how="outer", suffixes=("_fb", ""))
        result = merged[["frame_idx"]].copy()

        main_cols = [c for c in main.columns if c != "frame_idx"]
        fb_cols = [c for c in fb.columns if c != "frame_idx"]
        all_names = sorted(set(main_cols) | set(fb_cols))

        for name in all_names:
            col_main = name if name in merged.columns else None
            col_fb = f"{name}_fb" if f"{name}_fb" in merged.columns else None

            if col_main and col_fb:
                s = merged[col_main].where(merged[col_main].notna(), merged[col_fb])
                result[name] = s
            elif col_main:
                result[name] = merged[col_main]
            elif col_fb:
                result[name] = merged[col_fb]
        return result

    max_len = max(len(main), len(fb))
    main_ix = main.reindex(range(max_len)).reset_index(drop=True)
    fb_ix = fb.reindex(range(max_len)).reset_index(drop=True)
    result = pd.DataFrame(index=range(max_len))

    if "frame_idx" in main.columns:
        result["frame_idx"] = main_ix["frame_idx"]
    elif "frame_idx" in fb.columns:
        result["frame_idx"] = fb_ix["frame_idx"]
    else:
        result["frame_idx"] = np.arange(max_len)

    for name in sorted(set(main.columns) | set(fb.columns)):
        if name == "frame_idx":
            continue
        if name in main_ix.columns and name in fb_ix.columns:
            result[name] = main_ix[name].where(main_ix[name].notna(), fb_ix[name])
        elif name in main_ix.columns:
            result[name] = main_ix[name]
        else:
            result[name] = fb_ix[name]
    return result


def load_per_frame_preferring_csv(run_dir: Path) -> pd.DataFrame:
    main = _read_csv_safe(run_dir / "per_frame.csv")
    fb = _read_csv_safe(run_dir / "per_frame_recomputed.csv")
    return _align_fill(main, fb)


def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _to_bool_or_none(x):
    if pd.isna(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer, float, np.floating)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in ("true", "t", "yes", "y", "1", "on"):
        return True
    if s in ("false", "f", "no", "n", "0", "off"):
        return False
    return None


def fmt_pm(mean: float, std: float, digits: int) -> str:
    if pd.isna(mean) or pd.isna(std):
        return ""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def fmt_pm_mse(mean: float, std: float) -> str:
    if pd.isna(mean) or pd.isna(std):
        return ""
    return f"{mean:.3f} ± {std:.3f}" if abs(mean) >= 1e-2 else f"{mean:.2e} ± {std:.2e}"


def sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in s)


def metric_sort_key(df: pd.DataFrame) -> tuple[list[str], list[bool]]:
    cols, asc = [], []
    if "ssim_mean" in df.columns:
        cols.append("ssim_mean"); asc.append(False)
    if "mse_mean" in df.columns:
        cols.append("mse_mean");  asc.append(True)
    if "rt_mean" in df.columns:
        cols.append("rt_mean");   asc.append(True)
    if not cols:
        cols, asc = (["experiments"], [False]) if "experiments" in df.columns else ([], [])
    return cols, asc


def _get_any(d: dict, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


# ----------------- mapping config.json -> cfg.* -----------------
KNOWN_CONFIG_KEYS = {
    "gaussian_filtered": "cfg.gaussian_filtered",
    "diff_warp": "cfg.diff_warp",
    "visibility": "cfg.visibility",
    "template_strategy": "cfg.template_strategy",
    "grid_size": "cfg.grid_size",
    "method": "cfg.method",
}

def extract_cfg_from_config_json(cfgj: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(cfgj, dict):
        return out

    for k, dst in KNOWN_CONFIG_KEYS.items():
        if k in cfgj and not isinstance(cfgj[k], dict):
            out[dst] = cfgj[k]

    for k, v in cfgj.items():
        if k in ("data", "run"):
            continue
        if k in KNOWN_CONFIG_KEYS:
            continue
        if not isinstance(v, dict) and not isinstance(v, list):
            out[f"cfg.{k}"] = v

    return out


# ----------------- scanning + enrichment -----------------
def _resolve_path(candidate: str | None, base_dir: Path) -> Path | None:
    if not candidate:
        return None
    p = Path(candidate)
    if p.is_file() or p.is_dir():
        return p
    p2 = (base_dir / p).resolve()
    if p2.is_file() or p2.is_dir():
        return p2
    p3 = (base_dir.parent.parent / p).resolve()
    return p3 if (p3.is_file() or p3.is_dir()) else None


def _find_artifact_for_video(run_dir: Path) -> Path | None:
    """Only accept .mp4 or .tif/.tiff artifacts; explicitly skip any file named 'video.mp4'."""
    art = run_dir / "artifacts"
    mp4s = [p for p in art.glob("*.mp4") if p.is_file() and p.name.lower() != "video.mp4"]
    if mp4s:
        return sorted(mp4s)[0]
    tifs = [p for p in list(art.glob("*.tif")) + list(art.glob("*.tiff")) if p.is_file()]
    if tifs:
        return sorted(tifs)[0]
    return None


def _to_gray_float01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3]
        w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
        arr = (arr * w).sum(axis=-1)
    arr = arr.astype(np.float64, copy=False)
    vmax = float(arr.max()) if arr.size else 1.0
    if vmax > 1.0:
        try:
            info_max = float(np.iinfo(arr.dtype).max)
        except Exception:
            info_max = vmax
        if info_max == 0:
            info_max = 1.0
        arr = arr / info_max
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _corr_with_temporal_mean(frames: list[np.ndarray]) -> tuple[float, float]:
    if not frames:
        return (np.nan, np.nan)
    gs = [_to_gray_float01(f) for f in frames]
    h = min(g.shape[-2] for g in gs)
    w = min(g.shape[-1] for g in gs)
    def crop(a):
        hh, ww = a.shape[-2], a.shape[-1]
        top, left = (hh - h)//2, (ww - w)//2
        return a[..., top:top+h, left:left+w]
    gs = [crop(g) for g in gs]
    mean_img = np.mean(np.stack(gs, axis=0), axis=0)
    mean_flat = mean_img.ravel()
    mf_std = mean_flat.std()
    if mf_std <= 0:
        return (np.nan, np.nan)

    vals = []
    for g in gs:
        v = g.ravel()
        v_std = v.std()
        if v_std <= 0:
            vals.append(np.nan)
            continue
        cov = np.mean((v - v.mean()) * (mean_flat - mean_flat.mean()))
        corr = cov / (v_std * mf_std)
        vals.append(float(corr))
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy()
    if arr.size == 0:
        return (np.nan, np.nan)
    return (float(arr.mean()), float(arr.std(ddof=0)))


def compute_input_crispness_from_config(cfgj: dict, run_dir: Path) -> float:
    data = cfgj.get("data") if isinstance(cfgj, dict) else None
    in_path = None
    if isinstance(data, dict):
        in_path = data.get("path") or data.get("input") or data.get("video")

    in_file = _resolve_path(in_path, run_dir)
    if in_file is None:
        return np.nan

    try:
        _, frames, _ = load_video(str(in_file))
        if not frames:
            return np.nan
        vals = []
        for f in frames:
            try:
                vals.append(float(crispness(f)))
            except Exception:
                continue
        return float(np.mean(vals)) if vals else np.nan
    except Exception:
        return np.nan


def extract_metrics_from_result_json(resj: dict) -> dict[str, float]:
    """Pull metrics from result.json:
       - metrics.summary.{mse_mean, mse_std, crispness_improvement}
       - top-level runtime_s
    """
    out = {
        "res.mse_mean": np.nan,
        "res.mse_std": np.nan,
        "res.crispness_improvement": np.nan,
        "runtime_s": np.nan,
    }
    try:
        summ = (resj.get("metrics") or {}).get("summary") or {}
        out["res.mse_mean"] = _safe_float(summ.get("mse_mean"))
        out["res.mse_std"] = _safe_float(summ.get("mse_std"))
        out["res.crispness_improvement"] = _safe_float(summ.get("crispness_improvement"))
        out["runtime_s"] = _safe_float(resj.get("runtime_s"))
    except Exception:
        pass
    return out


def scan_runs_tree(runs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not runs_root.exists():
        raise SystemExit(f"Runs root '{runs_root}' does not exist.")

    for group_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        for exp_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()]):
            for cat_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
                for vid_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()]):
                    for run_dir in sorted(vid_dir.glob("run_*")):
                        try:
                            resj = load_result_json(run_dir)
                            cfgj = load_config_json(run_dir)

                            res_flat = flatten_dict(resj)

                            # cfg from result.json (flattened)
                            cfg_flat: dict[str, Any] = {}
                            for key in ("cfg", "config", "config_dict"):
                                if isinstance(resj.get(key), dict):
                                    cfg_flat.update(flatten_dict(resj[key], prefix="cfg"))
                            for k, v in list(resj.items()):
                                if k.startswith("cfg.") and k not in cfg_flat:
                                    cfg_flat[k] = v

                            # cfg from config.json (mapped to cfg.*) — only used to fill missing
                            cfg_from_config = extract_cfg_from_config_json(cfgj)
                            for k, v in cfg_from_config.items():
                                if k not in cfg_flat:
                                    cfg_flat[k] = v

                            module = resj.get("module", resj.get("algo", "-"))
                            ok = bool(resj.get("ok", True))

                            # Compute corr_with_mean from artifact (.mp4/.tif only; skip video.mp4)
                            cwm_mean, cwm_std = (np.nan, np.nan)
                            art_path = _find_artifact_for_video(run_dir)
                            if art_path is not None:
                                try:
                                    _, frames, _ = load_video(str(art_path))
                                    if frames:
                                        cwm_mean, cwm_std = _corr_with_temporal_mean(frames)
                                except Exception:
                                    pass

                            metrics_from_res = extract_metrics_from_result_json(resj)

                            row = {
                                "group": group_dir.name,
                                "exp_name": exp_dir.name,
                                "data.category": cat_dir.name,
                                "data.video_id": vid_dir.name,
                                "run_id": run_dir.name.replace("run_", ""),
                                "module": module if module is not None else "-",
                                "ok": ok,

                                # per-frame metrics will be filled later
                                "pf.ssim_mean": np.nan,
                                "pf.ssim_std":  np.nan,
                                "pf.mse_mean":  np.nan,
                                "pf.mse_std":   np.nan,

                                # crispness placeholders (per-run means)
                                CRISPNESS_CANON_COL: np.nan,
                                CRISPNESS_INPUT_COL: np.nan,

                                # corr with mean (from artifact video)
                                "m.corr_with_mean_mean": _safe_float(cwm_mean),
                                "m.corr_with_mean_std":  _safe_float(cwm_std),

                                # result.json metrics + runtime
                                "res.mse_mean": metrics_from_res["res.mse_mean"],
                                "res.mse_std": metrics_from_res["res.mse_std"],
                                "res.crispness_improvement": metrics_from_res["res.crispness_improvement"],
                                "runtime_s": metrics_from_res["runtime_s"],
                            }
                            row.update(cfg_flat)
                            rows.append(row)
                        except Exception:
                            continue
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No runs found under runs/ .")
    return df

def _read_aci_means(run_dir: Path) -> tuple[float, float]:
    """
    Read artifact_crispness_index.csv and return (out_mean, in_mean) as floats or NaN.
    """
    aci = _read_csv_safe(run_dir / "artifact_crispness_index.csv")
    if aci.empty:
        return (np.nan, np.nan)

    out_mean = np.nan
    in_mean = np.nan

    # Output crispness candidates
    for c in ("crispness", "crispness_out", "crisp_out", "m_crispness"):
        if c in aci.columns:
            vals = pd.to_numeric(aci[c], errors="coerce").dropna()
            if len(vals) > 0:
                out_mean = float(vals.mean())
                break

    # Input crispness candidates (optional, if your ACI has them)
    for c in ("crispness_input", "crispness_in", "crispness_raw", "raw_crispness", "input_crispness", "crisp_in"):
        if c in aci.columns:
            vals = pd.to_numeric(aci[c], errors="coerce").dropna()
            if len(vals) > 0:
                in_mean = float(vals.mean())
                break

    return (out_mean, in_mean)

def enrich_with_per_frame(runs_root: Path, df: pd.DataFrame) -> pd.DataFrame:
    # Ensure target columns exist
    for c in ["pf.ssim_mean", "pf.ssim_std", "pf.mse_mean", "pf.mse_std"]:
        if c not in df.columns:
            df[c] = np.nan
    if CRISPNESS_CANON_COL not in df.columns:
        df[CRISPNESS_CANON_COL] = np.nan
    if CRISPNESS_INPUT_COL not in df.columns:
        df[CRISPNESS_INPUT_COL] = np.nan

    for idx, row in df.iterrows():
        run_dir = (
            runs_root
            / str(row["group"])
            / str(row["exp_name"])
            / str(row["data.category"])
            / str(row["data.video_id"])
            / f"run_{row['run_id']}"
        )

        module_name = str(row.get("module", "")).strip().lower()

        # --- PRIMARY path for NoRMCorre: artifact_crispness_index.csv ---
        if module_name == "normcorre":
            aci_out, aci_in = _read_aci_means(run_dir)
            if not np.isnan(aci_out):
                df.at[idx, CRISPNESS_CANON_COL] = aci_out
            if not np.isnan(aci_in):
                df.at[idx, CRISPNESS_INPUT_COL] = aci_in

        # --- Per-frame CSVs are still the primary source for SSIM/MSE;
        #     and remain a fallback for crispness when needed. ---
        pf = load_per_frame_preferring_csv(run_dir)
        if pf.empty:
            continue

        # SSIM over frames
        if "ssim" in pf.columns:
            s = pd.to_numeric(pf["ssim"], errors="coerce").dropna()
            if len(s) > 0:
                df.at[idx, "pf.ssim_mean"] = float(s.mean())
                df.at[idx, "pf.ssim_std"]  = float(s.std(ddof=0))

        # MSE over frames
        if "mse" in pf.columns:
            m = pd.to_numeric(pf["mse"], errors="coerce").dropna()
            if len(m) > 0:
                df.at[idx, "pf.mse_mean"] = float(m.mean())
                df.at[idx, "pf.mse_std"]  = float(m.std(ddof=0))

        # Output crispness from per_frame (fallback for all; or primary for non-NoRMCorre)
        if pd.isna(df.at[idx, CRISPNESS_CANON_COL]) or module_name != "normcorre":
            for c_src, c_dst in (("crispness_out", CRISPNESS_CANON_COL),
                                 ("crispness", CRISPNESS_CANON_COL),
                                 ("crisp_out", CRISPNESS_CANON_COL)):
                if pd.isna(df.at[idx, c_dst]) and (c_src in pf.columns):
                    vals = pd.to_numeric(pf[c_src], errors="coerce").dropna()
                    if len(vals) > 0:
                        df.at[idx, c_dst] = float(vals.mean())

        # Input crispness from per_frame (fallback for all)
        if pd.isna(df.at[idx, CRISPNESS_INPUT_COL]):
            for c_src, c_dst in (("crispness_input", CRISPNESS_INPUT_COL),
                                 ("crispness_in", CRISPNESS_INPUT_COL),
                                 ("crispness_raw", CRISPNESS_INPUT_COL),
                                 ("raw_crispness", CRISPNESS_INPUT_COL),
                                 ("input_crispness", CRISPNESS_INPUT_COL),
                                 ("crisp_in", CRISPNESS_INPUT_COL)):
                if c_src in pf.columns:
                    vals = pd.to_numeric(pf[c_src], errors="coerce").dropna()
                    if len(vals) > 0:
                        df.at[idx, c_dst] = float(vals.mean())

    return df


def unify_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    # SSIM from per_frame
    df["m.ssim_mean"] = df["pf.ssim_mean"]
    df["m.ssim_std"]  = df["pf.ssim_std"]

    # MSE: prefer result.json summary if present; else per_frame
    df["m.mse_mean"] = df["res.mse_mean"].where(df["res.mse_mean"].notna(), df["pf.mse_mean"])
    df["m.mse_std"]  = df["res.mse_std"].where(df["res.mse_std"].notna(), df["pf.mse_std"])

    # Crispness improvement (as reported by result.json summary)
    df["m.crispness_improvement"] = df.get("res.crispness_improvement", np.nan)

    # runtime_s already present (top-level)
    return df


def _looks_like_identity(row: pd.Series) -> bool:
    m = str(row.get("cfg.method", "")).lower()
    if any(tok in m for tok in ["identity", "input", "raw", "uncorrected", "none", "baseline_input"]):
        return True
    gf = row.get("cfg.gaussian_filtered", None)
    dw = row.get("cfg.diff_warp", None)
    vis = row.get("cfg.visibility", None)
    def _false_or_none(x):
        return (x is None) or (isinstance(x, (bool, np.bool_)) and (not x)) or (str(x).lower() in ("false", "0", "off", "none", "nan"))
    if _false_or_none(gf) and _false_or_none(dw) and _false_or_none(vis):
        return True
    return False


def infer_missing_input_crispness(df: pd.DataFrame) -> pd.DataFrame:
    # Optional: try to infer input crispness from identity-like runs (kept for reference)
    key_vid = ["group", "module", "data.video_id"]
    df["_is_identity"] = df.apply(_looks_like_identity, axis=1)

    inferred = []
    for keys, g in df.groupby(key_vid, dropna=False):
        g = g.copy()
        ident = g[g["_is_identity"] & g[CRISPNESS_CANON_COL].notna()]
        val = np.nan
        if not ident.empty:
            val = float(pd.to_numeric(ident[CRISPNESS_CANON_COL], errors="coerce").dropna().median())
        if not np.isnan(val):
            d = dict(zip(key_vid, keys, strict=False))
            d["inferred_input_crispness"] = val
            inferred.append(d)

    if inferred:
        inf_df = pd.DataFrame(inferred)
        df = df.merge(inf_df, on=key_vid, how="left")
        if "inferred_input_crispness" in df.columns:
            df[CRISPNESS_INPUT_COL] = df[CRISPNESS_INPUT_COL].where(df[CRISPNESS_INPUT_COL].notna(), df["inferred_input_crispness"])
            df.drop(columns=["inferred_input_crispness"], inplace=True, errors="ignore")

    df.drop(columns=["_is_identity"], inplace=True, errors="ignore")
    return df


def normalize_flags_and_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if "cfg.gaussian_filtered" not in df.columns and "cfg.gaussian_filtering" in df.columns:
        df["cfg.gaussian_filtered"] = df["cfg.gaussian_filtering"]
    for flag in ["cfg.gaussian_filtered", "cfg.diff_warp", "cfg.visibility"]:
        if flag in df.columns:
            df[flag] = df[flag].apply(_to_bool_or_none)
    return df


# ----------------- parameter selection & aggregation -----------------
def select_param_columns_for_method(df_m: pd.DataFrame) -> list[str]:
    chosen: list[str] = [c for c in REQUIRED_PARAM_COLS if c in df_m.columns]

    cfg_cols = [c for c in df_m.columns if c.startswith("cfg.")]
    for c in cfg_cols:
        if c in chosen:
            continue
        vals = df_m[c].dropna().unique()
        nunique = len(vals)
        if nunique <= 1:
            continue
        is_boolish = df_m[c].dropna().map(lambda x: isinstance(x, (bool, np.bool_)) or str(x).lower() in ("true","false")).all()
        is_numeric = pd.api.types.is_numeric_dtype(df_m[c])
        if is_boolish or (is_numeric and nunique <= MAX_NUMERIC_UNIQUE) or (not is_numeric and nunique <= MAX_PARAM_UNIQUE):
            chosen.append(c)
        if len(chosen) >= MAX_TOTAL_PARAM_COLS:
            break

    seen, final = set(), []
    for c in chosen:
        if c not in seen:
            final.append(c); seen.add(c)
    return final


def per_experiment_means(df: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    """
    Produce ONE row per (group, module, exp_name, cfg...) by averaging over videos (and runs) inside that experiment.
    """
    keys = ["group", "module", "exp_name"] + param_cols
    agg = {
        "m.ssim_mean": ("m.ssim_mean", "mean"),
        "m.ssim_std":  ("m.ssim_std",  "mean"),   # average frame-stds within experiment
        "m.mse_mean":  ("m.mse_mean",  "mean"),
        "m.mse_std":   ("m.mse_std",   "mean"),
        "m.crispness_improvement": ("m.crispness_improvement", "mean"),
        CRISPNESS_CANON_COL: (CRISPNESS_CANON_COL, "mean"),
        CRISPNESS_INPUT_COL: (CRISPNESS_INPUT_COL, "mean"),
        "runtime_s": ("runtime_s", "mean"),
        "m.corr_with_mean_mean": ("m.corr_with_mean_mean", "mean"),
        "m.corr_with_mean_std":  ("m.corr_with_mean_std",  "mean"),
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}
    return df.groupby(keys, dropna=False).agg(**agg).reset_index()


def across_experiments_mean_std(exp_df: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate across experiments (exp_name) to get mean ± std per (group, module, cfg...).
    """
    keys = ["group", "module"] + param_cols
    metric_cols = [c for c in [
        "m.ssim_mean", "m.ssim_std",
        "m.mse_mean",  "m.mse_std",
        "m.crispness_improvement",
        CRISPNESS_CANON_COL, CRISPNESS_INPUT_COL,
        "runtime_s",
        "m.corr_with_mean_mean", "m.corr_with_mean_std",
    ] if c in exp_df.columns]

    agg = {}
    for c in metric_cols:
        agg[f"{c}_mean"] = (c, "mean")
        agg[f"{c}_std"]  = (c, "std")

    out = exp_df.groupby(keys, dropna=False).agg(**agg).reset_index()
    out["experiments"] = exp_df.groupby(keys, dropna=False)["exp_name"].nunique().values
    return out


def summarize_method(df_m: pd.DataFrame, runs_root: Path, group: str, module: str) -> pd.DataFrame:
    param_cols = select_param_columns_for_method(df_m)

    work = df_m.copy()
    # ensure numeric
    for col in [
        "m.ssim_mean","m.ssim_std",
        "m.mse_mean","m.mse_std",
        "m.crispness_improvement",
        CRISPNESS_CANON_COL, CRISPNESS_INPUT_COL,
        "runtime_s", "m.corr_with_mean_mean", "m.corr_with_mean_std"
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    # 1) per-experiment means (average inside experiment)
    exp_df = per_experiment_means(work, param_cols)

    # 2) across-experiments mean & std per configuration
    summ = across_experiments_mean_std(exp_df, param_cols)

    # Pretty-print columns
    if "m.ssim_mean_mean" in summ.columns and "m.ssim_mean_std" in summ.columns:
        summ["ssim_pm"] = [fmt_pm(m, s, 3) for m, s in zip(summ["m.ssim_mean_mean"], summ["m.ssim_mean_std"], strict=False)]
    if "m.mse_mean_mean" in summ.columns and "m.mse_mean_std" in summ.columns:
        summ["mse_pm"]  = [fmt_pm_mse(m, s) for m, s in zip(summ["m.mse_mean_mean"], summ["m.mse_mean_std"], strict=False)]
    if f"{CRISPNESS_CANON_COL}_mean" in summ.columns and f"{CRISPNESS_CANON_COL}_std" in summ.columns:
        summ["crisp_pm"]= [fmt_pm(m, s, 3) for m, s in zip(summ[f"{CRISPNESS_CANON_COL}_mean"], summ[f"{CRISPNESS_CANON_COL}_std"], strict=False)]
    if "runtime_s_mean" in summ.columns and "runtime_s_std" in summ.columns:
        summ["rt_pm"]   = [fmt_pm(m, s, 1) for m, s in zip(summ["runtime_s_mean"], summ["runtime_s_std"], strict=False)]

    # Sorting preference
    fake_for_sort = pd.DataFrame({
        "ssim_mean": summ.get("m.ssim_mean_mean", pd.Series(dtype=float)),
        "mse_mean":  summ.get("m.mse_mean_mean",  pd.Series(dtype=float)),
        "rt_mean":   summ.get("runtime_s_mean",   pd.Series(dtype=float)),
        "experiments": summ.get("experiments", pd.Series(dtype=float)),
    })
    sort_cols, sort_asc = metric_sort_key(fake_for_sort)
    mapped_sort_cols = []
    for c in sort_cols:
        if c == "ssim_mean": mapped_sort_cols.append("m.ssim_mean_mean")
        elif c == "mse_mean": mapped_sort_cols.append("m.mse_mean_mean")
        elif c == "rt_mean":  mapped_sort_cols.append("runtime_s_mean")
        elif c == "experiments": mapped_sort_cols.append("experiments")
    if mapped_sort_cols:
        summ = summ.sort_values(mapped_sort_cols, ascending=sort_asc).reset_index(drop=True)
        summ["rank_primary"] = np.arange(1, len(summ)+1)
    else:
        summ["rank_primary"] = np.nan

    # Cast booleans to strings for readability
    for c in param_cols:
        if c in summ.columns:
            if (summ[c].dtype == bool) or summ[c].dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all():
                summ[c] = summ[c].map({True: "True", False: "False"}).astype("object")

    # Choose output columns
    cols = ["group", "module"] + param_cols + ["experiments"]
    cols += [x for x in [
        "m.ssim_mean_mean", "m.ssim_mean_std", "ssim_pm",
        "m.mse_mean_mean",  "m.mse_mean_std",  "mse_pm",
        f"{CRISPNESS_CANON_COL}_mean", f"{CRISPNESS_CANON_COL}_std", "crisp_pm",
        "m.crispness_improvement_mean", "m.crispness_improvement_std",
        "runtime_s_mean", "runtime_s_std", "rt_pm",
        "m.corr_with_mean_mean_mean", "m.corr_with_mean_std_mean",
        "rank_primary",
    ] if x in summ.columns]
    cols = [c for c in cols if c in summ.columns]
    summ = summ[cols]

    out_csv = runs_root / f"paper_{sanitize_filename(group)}_{sanitize_filename(module)}.csv"
    summ.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  (rows={len(summ)}, params={param_cols})")
    return summ


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs", help="root folder containing run directories")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)

    df = scan_runs_tree(runs_root)

    # Only successful runs
    if "ok" in df.columns:
        df = df[df["ok"] == True]
    if df.empty:
        raise SystemExit("No successful runs to analyze (ok==True).")

    # Fill from per_frame
    df = enrich_with_per_frame(runs_root, df)

    # Normalize flags and unify metrics
    df = normalize_flags_and_aliases(df)
    df = unify_metric_columns(df)

    # (Optional) infer input crispness for reference if missing (doesn't affect improvement from result.json)
    #df = infer_missing_input_crispness(df)

    methods = df[["group", "module"]].drop_duplicates().sort_values(["group", "module"]).to_records(index=False)

    combined = []
    for group, module in methods:
        df_m = df[(df["group"] == group) & (df["module"] == module)].copy()
        if df_m.empty:
            continue
        summ = summarize_method(df_m, runs_root, group, module)
        if not summ.empty:
            combined.append(summ)

    if combined:
        all_methods = pd.concat(combined, ignore_index=True, sort=False)
        out_all = runs_root / "paper_all_methods.csv"
        all_methods.to_csv(out_all, index=False)
        print(f"\nCombined table saved: {out_all}")
    else:
        print("No per-method summaries produced (check runs/ structure and result.json contents).")


if __name__ == "__main__":
    main()
