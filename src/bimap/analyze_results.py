#!/usr/bin/env python3
"""
paper_summaries_from_config.py

Paper-ready tables with ONE ROW PER CONFIGURATION (read from config.json), averaged over videos.

What it does
------------
1) Walks runs/ tree (no table.csv).
2) For each run, reads:
   - config.json  -> parameters (mapped into cfg.* columns)
   - result.json  -> metrics (and cfg.* if present, used to fill gaps)
   - per_frame.*  -> fills missing SSIM if needed
3) Averages metrics per video (to avoid multiple-runs bias), then aggregates across videos
   per (group, module, configuration) to produce mean ± std.
4) Writes one CSV per method and a combined CSV.

Outputs
-------
- runs/paper_<group>_<module>.csv
- runs/paper_all_methods.csv

Config mapping
--------------
From config.json (top level or under known places) to cfg.*:
- gaussian_filtered      -> cfg.gaussian_filtered (bool)
- diff_warp              -> cfg.diff_warp (bool)
- visibility             -> cfg.visibility (bool)
- template_strategy      -> cfg.template_strategy (str)
- grid_size              -> cfg.grid_size (int/float)
- method                 -> cfg.method (str)

Any other scalar top-level keys in config.json are added as cfg.<key> as well.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ----------------- config -----------------
METRICS = [
    ("m.ssim_mean", "ssim"),
    ("m.mse_mean", "mse"),
    ("m.crispness_improvement", "crisp"),
]
RUNTIME = "runtime_s"

# Columns we always try to include in the per-method tables (if present)
REQUIRED_PARAM_COLS = [
    "cfg.template_strategy",
    "cfg.gaussian_filtered",
    "cfg.diff_warp",
    "cfg.visibility",
    "cfg.grid_size",
    "cfg.method",
]

# Limits for adding extra cfg.* params (to keep tables tidy)
MAX_PARAM_UNIQUE = 10
MAX_NUMERIC_UNIQUE = 8
MAX_TOTAL_PARAM_COLS = 12  # INCLUDING required params


# ----------------- helpers -----------------
def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
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
    # Your example uses config.json with keys like gaussian_filtered at top-level.
    return load_json(run_dir / "config.json")


def load_per_frame(run_dir: Path) -> pd.DataFrame:
    p1 = run_dir / "per_frame.parquet"
    p2 = run_dir / "per_frame.csv"
    if p1.exists():
        try:
            return pd.read_parquet(p1)
        except Exception:
            pass
    if p2.exists():
        try:
            return pd.read_csv(p2)
        except Exception:
            pass
    return pd.DataFrame({"frame_idx": []})


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


def metric_sort_key(df: pd.DataFrame) -> Tuple[List[str], List[bool]]:
    cols, asc = [], []
    if "ssim_mean" in df.columns:
        cols.append("ssim_mean"); asc.append(False)
    if "mse_mean" in df.columns:
        cols.append("mse_mean");  asc.append(True)
    if "rt_mean" in df.columns:
        cols.append("rt_mean");   asc.append(True)
    if not cols:
        cols, asc = (["videos"], [False]) if "videos" in df.columns else ([], [])
    return cols, asc


# ----------------- mapping config.json -> cfg.* -----------------
KNOWN_CONFIG_KEYS = {
    "gaussian_filtered": "cfg.gaussian_filtered",
    "diff_warp": "cfg.diff_warp",
    "visibility": "cfg.visibility",
    "template_strategy": "cfg.template_strategy",
    "grid_size": "cfg.grid_size",
    "method": "cfg.method",
}

def extract_cfg_from_config_json(cfgj: dict) -> Dict[str, Any]:
    """
    Pull known scalar keys from config.json into cfg.* namespace.
    Also copy any *other* scalar top-level keys as cfg.<key>.
    """
    out: Dict[str, Any] = {}
    if not isinstance(cfgj, dict):
        return out

    # Known keys
    for k, dst in KNOWN_CONFIG_KEYS.items():
        if k in cfgj and not isinstance(cfgj[k], dict):
            out[dst] = cfgj[k]

    # Other scalar top-level keys -> cfg.<key>
    for k, v in cfgj.items():
        if k in ("data", "run"):  # skip nested metadata blocks
            continue
        if k in KNOWN_CONFIG_KEYS:
            continue
        if not isinstance(v, dict) and not isinstance(v, list):
            out[f"cfg.{k}"] = v

    return out


# ----------------- scanning + enrichment -----------------
def scan_runs_tree(runs_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
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

                            # cfg from result.json (flattened)
                            cfg_flat: Dict[str, Any] = {}
                            for key in ("cfg", "config", "config_dict"):
                                if isinstance(resj.get(key), dict):
                                    cfg_flat.update(flatten_dict(resj[key], prefix="cfg"))
                            for k, v in list(resj.items()):
                                if k.startswith("cfg.") and k not in cfg_flat:
                                    cfg_flat[k] = v

                            # cfg from config.json (mapped to cfg.*)
                            cfg_from_config = extract_cfg_from_config_json(cfgj)
                            # Merge: result.json cfg wins if both present, otherwise take from config.json
                            for k, v in cfg_from_config.items():
                                if k not in cfg_flat:
                                    cfg_flat[k] = v

                            metrics_summary = ((resj.get("metrics") or {}).get("summary") or {})
                            module = resj.get("module", resj.get("algo", "-"))
                            ok = bool(resj.get("ok", True))

                            row = {
                                "group": group_dir.name,
                                "exp_name": exp_dir.name,
                                "data.category": cat_dir.name,
                                "data.video_id": vid_dir.name,
                                "run_id": run_dir.name.replace("run_", ""),
                                "module": module if module is not None else "-",
                                "ok": ok,
                                "m.mse_mean": _safe_float(metrics_summary.get("mse_mean")),
                                "m.crispness_improvement": _safe_float(metrics_summary.get("crispness_improvement")),
                                "runtime_s": _safe_float(resj.get("runtime_s")),
                            }
                            row.update(cfg_flat)

                            if "m.ssim_mean" in metrics_summary:
                                row["m.ssim_mean"] = _safe_float(metrics_summary.get("m.ssim_mean"))

                            rows.append(row)
                        except Exception:
                            continue
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No runs found under runs/ .")
    return df


def enrich_with_per_frame_ssim(runs_root: Path, df: pd.DataFrame) -> pd.DataFrame:
    if "m.ssim_mean" not in df.columns:
        df["m.ssim_mean"] = np.nan
    needs = df["m.ssim_mean"].isna()
    if needs.any():
        for idx, row in df[needs].iterrows():
            run_dir = runs_root / str(row["group"]) / str(row["exp_name"]) / str(row["data.category"]) / str(row["data.video_id"]) / f"run_{row['run_id']}"
            pf = load_per_frame(run_dir)
            if "ssim" in pf.columns and len(pf["ssim"]) > 0:
                ssim_mean = pd.to_numeric(pf["ssim"], errors="coerce").dropna().mean()
                if not np.isnan(ssim_mean):
                    df.at[idx, "m.ssim_mean"] = float(ssim_mean)
    return df


def normalize_flags_and_aliases(df: pd.DataFrame) -> pd.DataFrame:
    # Alias: if someone used cfg.gaussian_filtering somewhere, map to cfg.gaussian_filtered
    if "cfg.gaussian_filtered" not in df.columns and "cfg.gaussian_filtering" in df.columns:
        df["cfg.gaussian_filtered"] = df["cfg.gaussian_filtering"]
    # Normalize booleans
    for flag in ["cfg.gaussian_filtered", "cfg.diff_warp", "cfg.visibility"]:
        if flag in df.columns:
            df[flag] = df[flag].apply(_to_bool_or_none)
    return df


# ----------------- parameter selection & aggregation -----------------
def select_param_columns_for_method(df_m: pd.DataFrame) -> List[str]:
    chosen: List[str] = [c for c in REQUIRED_PARAM_COLS if c in df_m.columns]

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
        if is_boolish:
            chosen.append(c)
        elif is_numeric and nunique <= MAX_NUMERIC_UNIQUE:
            chosen.append(c)
        elif not is_numeric and nunique <= MAX_PARAM_UNIQUE:
            chosen.append(c)
        if len(chosen) >= MAX_TOTAL_PARAM_COLS:
            break

    # unique, stable
    seen, final = set(), []
    for c in chosen:
        if c not in seen:
            final.append(c); seen.add(c)
    return final


def per_video_means(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    agg_spec = {}
    for mcol, _short in METRICS:
        if mcol in df.columns:
            agg_spec[mcol] = (mcol, "mean")
    if RUNTIME in df.columns:
        agg_spec[RUNTIME] = (RUNTIME, "mean")
    return df.groupby(key_cols, dropna=False).agg(**agg_spec).reset_index()


def summarize_method(df_m: pd.DataFrame, runs_root: Path, group: str, module: str) -> pd.DataFrame:
    # ALWAYS include required params (if present), plus a few extras
    param_cols = select_param_columns_for_method(df_m)

    # ensure numeric
    work = df_m.copy()
    for c,_ in METRICS:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    if RUNTIME in work.columns:
        work[RUNTIME] = pd.to_numeric(work[RUNTIME], errors="coerce")

    # per-video means to prevent run-count bias
    key_video = ["group", "module", "data.video_id"] + param_cols
    pv = per_video_means(work, key_video)

    # aggregate across videos per configuration
    key_cfg = ["group", "module"] + param_cols
    agg = {}
    if "m.ssim_mean" in pv.columns:
        agg["ssim_mean"] = ("m.ssim_mean", "mean")
        agg["ssim_std"]  = ("m.ssim_mean", "std")
    if "m.mse_mean" in pv.columns:
        agg["mse_mean"]  = ("m.mse_mean", "mean")
        agg["mse_std"]   = ("m.mse_mean", "std")
    if "m.crispness_improvement" in pv.columns:
        agg["crisp_mean"] = ("m.crispness_improvement", "mean")
        agg["crisp_std"]  = ("m.crispness_improvement", "std")
    if RUNTIME in pv.columns:
        agg["rt_mean"]   = (RUNTIME, "mean")
        agg["rt_std"]    = (RUNTIME, "std")

    vids = pv.groupby(key_cfg, dropna=False)["data.video_id"].nunique().rename("videos").reset_index()
    summ = pv.groupby(key_cfg, dropna=False).agg(**agg).reset_index().merge(vids, on=key_cfg, how="left")

    # format mean ± std
    if "ssim_mean" in summ.columns:
        summ["ssim_pm"] = [fmt_pm(m, s, 3) for m,s in zip(summ["ssim_mean"], summ["ssim_std"])]
    if "mse_mean" in summ.columns:
        summ["mse_pm"]  = [fmt_pm_mse(m, s) for m,s in zip(summ["mse_mean"], summ["mse_std"])]
    if "crisp_mean" in summ.columns:
        summ["crisp_pm"]= [fmt_pm(m, s, 3) for m,s in zip(summ["crisp_mean"], summ["crisp_std"])]
    if "rt_mean" in summ.columns:
        summ["rt_pm"]   = [fmt_pm(m, s, 1) for m,s in zip(summ["rt_mean"], summ["rt_std"])]

    # sort and rank within method
    sort_cols, sort_asc = metric_sort_key(summ)
    if sort_cols:
        summ = summ.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)
        summ["rank_primary"] = np.arange(1, len(summ)+1)
    else:
        summ["rank_primary"] = np.nan

    # prettify booleans
    for c in param_cols:
        if c in summ.columns:
            if summ[c].dtype == bool or summ[c].dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all():
                summ[c] = summ[c].map({True: "True", False: "False"}).astype("object")

    # column order
    cols = ["group", "module"] + param_cols + ["videos"]
    for base in ("ssim", "mse", "crisp"):
        if f"{base}_mean" in summ.columns:
            cols += [f"{base}_mean", f"{base}_std", f"{base}_pm"]
    if "rt_mean" in summ.columns:
        cols += ["rt_mean", "rt_std", "rt_pm"]
    cols += ["rank_primary"]
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

    # Build dataframe
    df = scan_runs_tree(runs_root)
    df = enrich_with_per_frame_ssim(runs_root, df)

    # Keep only successful runs (if ok key is present)
    if "ok" in df.columns:
        df = df[df["ok"] == True]
    if df.empty:
        raise SystemExit("No successful runs to analyze (ok==True).")

    # Normalize/alias booleans
    df = normalize_flags_and_aliases(df)

    # Identify methods
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
        print("No per-method summaries produced (check runs/ structure and result.json/config.json contents).")


if __name__ == "__main__":
    main()
