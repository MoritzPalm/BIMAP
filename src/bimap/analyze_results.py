#!/usr/bin/env python3
"""
analyze_from_runs_csv.py — enrich runs.csv from result.json/per_frame and pick best settings.

Assumes run layout:
runs/
  <group>/<exp_name>/<category>/<video_id>/run_<run_id>/
    result.json
    per_frame.parquet (optional)
    per_frame.csv     (fallback)
    frames/           (optional; raw or processed frame images, any *.png/*.jpg)

Your runs.csv columns (as pasted):
group,exp_name,module,run_id,created_at,data.video_id,data.path,data.category,
cfg.template_strategy,ok,error,duration_s,cfg.gaussian_filtered

Outputs:
  runs/leaderboard_<metric>.csv
  prints the best algorithm + settings to stdout
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd

# Optional PIL import for image-based correlation; guarded at call sites.
try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

# -------- helpers --------
LOWER_IS_BETTER = ("loss", "mse", "mae", "rmse", "error", "epe", "ae", "nrmse")
HIGHER_IS_BETTER = ("ssim", "ncc", "psnr", "dice", "iou", "accuracy", "f1", "auc", "crispness", "mi", "corr")

def metric_direction(name: str) -> str:
    n = name.lower()
    if any(k in n for k in LOWER_IS_BETTER):
        return "lower"
    if any(k in n for k in HIGHER_IS_BETTER):
        return "higher"
    return "higher"

def pick_metric(cols: list[str]) -> str:
    prefs = ["m.ssim_mean", "m.corr_mean_image", "m.crispness_improvement", "m.mse_mean"]
    for p in prefs:
        if p in cols:
            return p
    # last resort: if nothing enriched, prefer duration (lower better)
    return "m.ssim_mean" if "m.ssim_mean" in cols else ("m.mse_mean" if "m.mse_mean" in cols else "duration_s")

def run_dir_from_row(runs_root: Path, row: pd.Series) -> Path:
    cat = (row.get("data.category") or "uncat")
    return (
        runs_root
        / str(row["group"])
        / str(row["exp_name"])
        / str(cat)
        / str(row["data.video_id"])
        / f"run_{row['run_id']}"
    )

def load_result_json(run_dir: Path) -> dict:
    p = run_dir / "result.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

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

def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation for 1D arrays; returns np.nan on degenerate vectors."""
    a = a.astype(np.float64, copy=False).ravel()
    b = b.astype(np.float64, copy=False).ravel()
    if a.size != b.size or a.size == 0:
        return np.nan
    a_m = a - a.mean()
    b_m = b - b.mean()
    denom = np.linalg.norm(a_m) * np.linalg.norm(b_m)
    if denom == 0:
        return np.nan
    return float(np.dot(a_m, b_m) / denom)

def _load_frames_from_dir(frames_dir: Path, limit: Optional[int] = 300) -> Optional[np.ndarray]:
    if Image is None:
        return None
    if not frames_dir.exists():
        return None
    files: Sequence[Path] = sorted([p for p in frames_dir.glob("*.png")] + [p for p in frames_dir.glob("*.jpg")])
    if not files:
        return None
    if limit is not None and len(files) > limit:
        # uniform sub-sample up to 'limit'
        idx = np.linspace(0, len(files) - 1, num=limit, dtype=int)
        files = [files[i] for i in idx]
    # Load as grayscale float32 in [0,1]
    imgs = []
    w = h = None
    for fp in files:
        try:
            im = Image.open(fp).convert("L")
            if w is None:
                w, h = im.size
            else:
                if im.size != (w, h):
                    im = im.resize((w, h))  # naive resize to align dims
            arr = np.asarray(im, dtype=np.float32) / 255.0
            imgs.append(arr)
        except Exception:
            continue
    if not imgs:
        return None
    stack = np.stack(imgs, axis=0)  # (T,H,W)
    return stack

def compute_corr_mean_image(run_dir: Path, pf: Optional[pd.DataFrame], resj: dict) -> Optional[float]:
    """Compute correlation with mean image (average across frames). Strategy:
       1) Use result.json -> metrics.summary.corr_mean_image if present.
       2) If per_frame has a correlation-to-mean column, average it.
       3) Else compute from frames/ by correlating each frame to the mean image.
    """
    # 1) result.json summary
    try:
        val = (resj.get("metrics") or {}).get("summary", {}).get("corr_mean_image", None)
        if val is not None:
            return float(val)
    except Exception:
        pass

    # 2) per_frame columns
    if pf is not None and not pf.empty:
        for cname in ["corr_to_mean", "corr_mean", "ncc_to_mean"]:
            if cname in pf.columns:
                s = pd.to_numeric(pf[cname], errors="coerce").dropna()
                if len(s) > 0:
                    return float(s.mean())

    # 3) frames/ directory
    frames_dir = run_dir / "frames"
    stack = _load_frames_from_dir(frames_dir, limit=300)
    if stack is None:
        return None
    # Compute mean image
    mean_img = stack.mean(axis=0)
    # Correlate each frame with mean image
    corrs = []
    m_flat = mean_img.ravel()
    for i in range(stack.shape[0]):
        c = _pearson_corr(stack[i].ravel(), m_flat)
        if not np.isnan(c):
            corrs.append(c)
    if not corrs:
        return None
    return float(np.mean(corrs))

def enrich_row_from_results(runs_root: Path, row: pd.Series) -> pd.Series:
    run_dir = run_dir_from_row(runs_root, row)
    resj = load_result_json(run_dir)

    # runtime_s
    rs = resj.get("runtime_s") if isinstance(resj, dict) else None
    if rs is not None:
        row["runtime_s"] = float(rs)

    # summary metrics (raw)
    summ = (resj.get("metrics") or {}).get("summary", {}) if isinstance(resj, dict) else {}
    if "mse_mean" in summ:
        row["m.mse_mean"] = float(summ["mse_mean"])
    if "mse_std" in summ:
        row["m.mse_std"] = float(summ["mse_std"])
    if "crispness_improvement" in summ:
        # keep raw for reference; percentage handling is done in improvement stage
        row["m.crispness_improvement"] = float(summ["crispness_improvement"])

    # per-frame enrichment
    pf = load_per_frame(run_dir)

    # ssim_mean (if not present) from per-frame
    if "m.ssim_mean" not in row or pd.isna(row.get("m.ssim_mean", np.nan)):
        if "ssim" in pf.columns and len(pf["ssim"]) > 0:
            row["m.ssim_mean"] = float(pd.to_numeric(pf["ssim"], errors="coerce").dropna().mean())

    # corr with mean image metric
    corr = compute_corr_mean_image(run_dir, pf, resj)
    if corr is not None:
        row["m.corr_mean_image"] = float(corr)

    return row

def _to_bool_or_none(x):
    if pd.isna(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in ("true", "t", "yes", "y", "1", "on"):
        return True
    if s in ("false", "f", "no", "n", "0", "off"):
        return False
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs", help="root folder that contains runs.csv and run directories")
    ap.add_argument("--metric", default="auto", help="primary metric (e.g., m.ssim_mean) or 'auto' (for ranking only)")
    ap.add_argument("--topn", type=int, default=10, help="top-N rows to print")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    # Prefer table.csv if present; else use runs.csv as provided
    csv_candidates = [runs_root / "table.csv", runs_root / "runs.csv"]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        raise SystemExit(f"Could not find {csv_candidates[0]} or {csv_candidates[1]}.")

    df_raw = pd.read_csv(csv_path)
    if df_raw.empty:
        raise SystemExit("runs.csv/table.csv is empty.")

    # Enrich each row from result.json/per_frame.*
    enriched = []
    for _, r in df_raw.iterrows():
        enriched.append(enrich_row_from_results(runs_root, r))
    df = pd.DataFrame(enriched)

    # Use only successful runs
    if "ok" in df.columns:
        df = df[df["ok"] == True]
    if df.empty:
        raise SystemExit("No successful runs to analyze (ok==True).")

    # Keep a copy with ALL metrics for improvement calculations
    df_all = df.copy()
    # Normalize boolean flag column (if present)
    if "cfg.gaussian_filtered" in df_all.columns:
        df_all["cfg.gaussian_filtered.bool"] = df_all["cfg.gaussian_filtered"].apply(_to_bool_or_none)

    # ---- Determine ranking metric (still one, just for leaderboard order) ----
    mcols = [c for c in df.columns if c.startswith("m.")]
    rank_metric = pick_metric(mcols) if args.metric == "auto" else args.metric
    if rank_metric not in df.columns:
        raise SystemExit(f"Metric '{rank_metric}' not found after enrichment. Available: {', '.join(mcols) or '(none)'}")
    direction = metric_direction(rank_metric)
    print(f"[info] ranking by '{rank_metric}' (direction={direction})")

    # Filter for ranking computations only
    dfr = df[pd.notna(df[rank_metric])].copy()
    dfr[rank_metric] = pd.to_numeric(dfr[rank_metric], errors="coerce")
    dfr = dfr[pd.notna(dfr[rank_metric])]
    if dfr.empty:
        raise SystemExit("No numeric values for ranking metric after cleaning.")

    # Rank per (video) so each video contributes fairly
    ascending = (direction == "lower")
    dfr["rank"] = dfr.groupby("data.video_id")[rank_metric].rank(method="average", ascending=ascending)

    # normalized score in [0,1] per video (higher=better)
    def norm_scores(g):
        v = g[rank_metric].astype(float).to_numpy()
        vmin, vmax = np.min(v), np.max(v)
        if np.isclose(vmin, vmax):
            g["score"] = 0.5
        else:
            g["score"] = (v - vmin) / (vmax - vmin) if direction == "higher" else (vmax - v) / (vmax - vmin)
        return g

    dfr = dfr.groupby("data.video_id", group_keys=False).apply(norm_scores)

    # ---- Aggregate by algorithm + settings (base leaderboard) ----
    key_cols = [
        "group",
        "exp_name",
        "module",
        "cfg.template_strategy" if "cfg.template_strategy" in dfr.columns else None,
        "cfg.gaussian_filtered" if "cfg.gaussian_filtered" in dfr.columns else None,
    ]
    key_cols = [k for k in key_cols if k is not None]

    agg_dict = dict(
        mean_score=("score", "mean"),
        mean_rank=("rank", "mean"),
        videos=("data.video_id", "nunique"),
        runs=("data.video_id", "count"),
    )
    if "runtime_s" in dfr.columns:
        agg_dict["mean_runtime"] = ("runtime_s", "mean")
    if "m.ssim_mean" in dfr.columns:
        agg_dict["m_ssim_mean"] = ("m.ssim_mean", "mean")
    if "m.mse_mean" in dfr.columns:
        agg_dict["m_mse_mean"] = ("m.mse_mean", "mean")
    if "m.crispness_improvement" in dfr.columns:
        # keep the raw average for reference; % improvement is reported separately
        agg_dict["m_crispness_raw"] = ("m.crispness_improvement", "mean")
    if "m.corr_mean_image" in dfr.columns:
        agg_dict["m_corr_mean_image"] = ("m.corr_mean_image", "mean")

    agg = dfr.groupby(key_cols, dropna=False).agg(**agg_dict).reset_index()

    # ---- Compute improvements vs baseline for ALL metrics on df_all (independent of ranking metric) ----
    # We compute:
    #  - SSIM: absolute delta (higher is better)
    #  - MSE: absolute delta (lower is better; we invert)
    #  - CORR: absolute delta (higher is better)
    #  - CRISPNESS: *percentage* change vs baseline, as requested.
    metrics_info_abs = [
        ("m.ssim_mean", "ssim"),
        ("m.mse_mean", "mse"),
        ("m.corr_mean_image", "corr"),
    ]
    # Experiment identity (excluding filtered flag) for baseline matching
    base_key_cols = ["group", "exp_name", "module", "data.video_id"]
    if "cfg.template_strategy" in df_all.columns:
        base_key_cols.append("cfg.template_strategy")

    impr_aggs = []  # will collect per-metric aggregated improvements to merge with leaderboard

    if "cfg.gaussian_filtered.bool" in df_all.columns:
        # Absolute-delta metrics (SSIM, MSE, CORR)
        for metric_col, short in metrics_info_abs:
            if metric_col not in df_all.columns:
                continue  # skip metrics that don't exist

            tmp = df_all.copy()
            tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")

            base_df = (
                tmp[(tmp["cfg.gaussian_filtered.bool"] == False) & pd.notna(tmp[metric_col])]
                [base_key_cols + [metric_col]]
                .rename(columns={metric_col: f"_{short}_baseline"})
            )

            tmp = tmp.merge(base_df, on=base_key_cols, how="left")
            tmp[f"_has_base_{short}"] = pd.notna(tmp[f"_{short}_baseline"])

            # improvement respecting direction (absolute delta)
            d = metric_direction(metric_col)
            if d == "higher":
                tmp[f"_impr_{short}"] = tmp[metric_col] - tmp[f"_{short}_baseline"]
            else:
                tmp[f"_impr_{short}"] = tmp[f"_{short}_baseline"] - tmp[metric_col]

            # For baseline rows (unfiltered) where baseline exists, set improvement to 0
            mask_baseline_rows = (tmp["cfg.gaussian_filtered.bool"] == False) & tmp[f"_has_base_{short}"]
            tmp.loc[mask_baseline_rows, f"_impr_{short}"] = 0.0

            # Aggregate per config
            cols_for_group = key_cols + [f"_impr_{short}"]
            tmp_group = tmp[cols_for_group + ["data.video_id", f"_has_base_{short}"]].copy()

            g_mean = (
                tmp_group.groupby(key_cols, dropna=False)[f"_impr_{short}"]
                .mean()
                .rename(f"avg_impr_vs_unfiltered_{short}")
            )
            vids_with_base = (
                tmp_group[tmp_group[f"_has_base_{short}"]]
                .groupby(key_cols, dropna=False)["data.video_id"]
                .nunique()
                .rename(f"videos_with_baseline_{short}")
            )
            impr_aggs.append(pd.concat([g_mean, vids_with_base], axis=1).reset_index())

        # CRISPNESS as *percentage* change vs baseline: 100 * (val - base) / |base|
        crisp_col = "m.crispness_improvement"
        if crisp_col in df_all.columns:
            tmp = df_all.copy()
            tmp[crisp_col] = pd.to_numeric(tmp[crisp_col], errors="coerce")

            base_df = (
                tmp[(tmp["cfg.gaussian_filtered.bool"] == False) & pd.notna(tmp[crisp_col])]
                [base_key_cols + [crisp_col]]
                .rename(columns={crisp_col: "_crisp_base"})
            )
            tmp = tmp.merge(base_df, on=base_key_cols, how="left")
            tmp["_has_base_crisp"] = pd.notna(tmp["_crisp_base"])

            # percentage change
            denom = tmp["_crisp_base"].abs()
            with np.errstate(divide="ignore", invalid="ignore"):
                tmp["_crisp_pct_change"] = 100.0 * (tmp[crisp_col] - tmp["_crisp_base"]) / denom
            # Handle divide-by-zero: if baseline is 0, set pct_change to NaN unless value also 0
            mask_zero_base = denom == 0
            tmp.loc[mask_zero_base & (tmp[crisp_col] == 0), "_crisp_pct_change"] = 0.0
            tmp.loc[mask_zero_base & (tmp[crisp_col] != 0), "_crisp_pct_change"] = np.nan

            # Baseline rows should be 0% change where baseline exists
            mask_baseline_rows = (tmp["cfg.gaussian_filtered.bool"] == False) & tmp["_has_base_crisp"]
            tmp.loc[mask_baseline_rows, "_crisp_pct_change"] = 0.0

            cols_for_group = key_cols + ["_crisp_pct_change"]
            tmp_group = tmp[cols_for_group + ["data.video_id", "_has_base_crisp"]].copy()

            g_mean = (
                tmp_group.groupby(key_cols, dropna=False)["_crisp_pct_change"]
                .mean()
                .rename("avg_pct_change_vs_unfiltered_crispness")
            )
            vids_with_base = (
                tmp_group[tmp_group["_has_base_crisp"]]
                .groupby(key_cols, dropna=False)["data.video_id"]
                .nunique()
                .rename("videos_with_baseline_crispness")
            )
            impr_aggs.append(pd.concat([g_mean, vids_with_base], axis=1).reset_index())

    # Merge all per-metric improvement summaries into leaderboard
    if impr_aggs:
        impr_agg = impr_aggs[0]
        for extra in impr_aggs[1:]:
            impr_agg = impr_agg.merge(extra, on=key_cols, how="outer")
        agg = agg.merge(impr_agg, on=key_cols, how="left")

    # ---- Order by mean_score desc, tie-breaker mean_rank asc ----
    agg = agg.sort_values(by=["mean_score", "mean_rank"], ascending=[False, True])

    # Save leaderboard
    out_csv = runs_root / f"leaderboard_{rank_metric.replace('.', '_')}.csv"
    agg.to_csv(out_csv, index=False)

    # Print a tidy top-N and the absolute best row
    print("\n=== TOP configs (by normalized score) ===")
    show_cols = [
        "group",
        "exp_name",
        "cfg.template_strategy",
        "cfg.gaussian_filtered",
        "mean_score",
        "mean_rank",
        "videos",
        "runs",
        "mean_runtime",
        "m_ssim_mean",
        "m_mse_mean",
        "m_corr_mean_image",
        "m_crispness_raw",  # raw average of the provided crispness metric (for reference)
        "avg_impr_vs_unfiltered_ssim",
        "videos_with_baseline_ssim",
        "avg_impr_vs_unfiltered_mse",
        "videos_with_baseline_mse",
        "avg_impr_vs_unfiltered_corr",
        "videos_with_baseline_corr",
        "avg_pct_change_vs_unfiltered_crispness",
        "videos_with_baseline_crispness",
    ]
    show = [c for c in show_cols if c in agg.columns]
    print(agg[show].head(args.topn).to_string(index=False))

    best = agg.iloc[0].to_dict()
    print("\n=== BEST overall ===")
    for k in show:
        if k in best:
            print(f"{k}: {best[k]}")
    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()
