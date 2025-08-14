#!/usr/bin/env python3
"""
analyze_from_runs_csv.py — enrich runs.csv from result.json/per_frame and pick best settings.

Assumes run layout:
runs/
  <group>/<exp_name>/<category>/<video_id>/run_<run_id>/
    result.json
    per_frame.parquet (optional)
    per_frame.csv     (fallback)

Your runs.csv columns (as pasted):
group,exp_name,module,run_id,created_at,data.video_id,data.path,data.category,
cfg.template_strategy,ok,error,duration_s,cfg.gaussian_filtered

Outputs:
  runs/leaderboard_<metric>.csv
  prints the best algorithm + settings to stdout
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# -------- helpers --------
LOWER_IS_BETTER = ("loss","mse","mae","rmse","error","epe","ae","nrmse")
HIGHER_IS_BETTER = ("ssim","ncc","psnr","dice","iou","accuracy","f1","auc","crispness","mi")

def metric_direction(name: str) -> str:
    n = name.lower()
    if any(k in n for k in LOWER_IS_BETTER):  return "lower"
    if any(k in n for k in HIGHER_IS_BETTER): return "higher"
    return "higher"

def pick_metric(cols: list[str]) -> str:
    prefs = ["m.ssim_mean", "m.crispness_improvement", "m.mse_mean"]
    for p in prefs:
        if p in cols: return p
    # last resort: if nothing enriched, prefer duration (lower better)
    return "m.ssim_mean" if "m.ssim_mean" in cols else ("m.mse_mean" if "m.mse_mean" in cols else "duration_s")

def run_dir_from_row(runs_root: Path, row: pd.Series) -> Path:
    cat = (row.get("data.category") or "uncat")
    return (runs_root / str(row["group"]) / str(row["exp_name"]) / str(cat) /
            str(row["data.video_id"]) / f"run_{row['run_id']}")

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
        try: return pd.read_parquet(p1)
        except Exception: pass
    if p2.exists():
        try: return pd.read_csv(p2)
        except Exception: pass
    return pd.DataFrame({"frame_idx": []})

def enrich_row_from_results(runs_root: Path, row: pd.Series) -> pd.Series:
    run_dir = run_dir_from_row(runs_root, row)
    resj = load_result_json(run_dir)

    # runtime_s
    rs = resj.get("runtime_s") if isinstance(resj, dict) else None
    if rs is not None:
        row["runtime_s"] = float(rs)

    # summary metrics
    summ = (resj.get("metrics") or {}).get("summary", {}) if isinstance(resj, dict) else {}
    if "mse_mean" in summ:
        row["m.mse_mean"] = float(summ["mse_mean"])
    if "mse_std" in summ:
        row["m.mse_std"] = float(summ["mse_std"])
    if "crispness_improvement" in summ:
        row["m.crispness_improvement"] = float(summ["crispness_improvement"])

    # ssim_mean (if not present) from per-frame
    if "m.ssim_mean" not in row or pd.isna(row.get("m.ssim_mean", np.nan)):
        pf = load_per_frame(run_dir)
        if "ssim" in pf.columns and len(pf["ssim"]) > 0:
            row["m.ssim_mean"] = float(pd.to_numeric(pf["ssim"], errors="coerce").dropna().mean())

    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs", help="root folder that contains runs.csv and run directories")
    ap.add_argument("--metric", default="auto", help="primary metric (e.g., m.ssim_mean) or 'auto'")
    ap.add_argument("--topn", type=int, default=10, help="top-N rows to print")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    # Prefer table.csv if present; else use runs.csv as provided
    csv_candidates = [runs_root / "table.csv", runs_root / "runs.csv"]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        raise SystemExit(f"Could not find {csv_candidates[0]} or {csv_candidates[1]}.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("runs.csv/table.csv is empty.")

    # Enrich each row from result.json/per_frame.*
    enriched = []
    for _, r in df.iterrows():
        enriched.append(enrich_row_from_results(runs_root, r))
    df = pd.DataFrame(enriched)

    # Use only successful runs
    if "ok" in df.columns:
        df = df[df["ok"] == True]
    if df.empty:
        raise SystemExit("No successful runs to analyze (ok==True).")

    # Choose metric
    mcols = [c for c in df.columns if c.startswith("m.")]
    metric = pick_metric(mcols) if args.metric == "auto" else args.metric
    if metric not in df.columns:
        raise SystemExit(f"Metric '{metric}' not found after enrichment. Available: {', '.join(mcols) or '(none)'}")
    direction = metric_direction(metric)
    print(f"[info] using metric '{metric}' (direction={direction})")

    # Rank per (video) so each video contributes fairly
    ascending = (direction == "lower")
    df = df[pd.notna(df[metric])]
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[pd.notna(df[metric])]
    if df.empty:
        raise SystemExit("No numeric metric values after cleaning.")

    df["rank"] = df.groupby("data.video_id")[metric].rank(method="average", ascending=ascending)

    # normalized score in [0,1] per video (higher=better)
    def norm_scores(g):
        v = g[metric].astype(float).to_numpy()
        vmin, vmax = np.min(v), np.max(v)
        if np.isclose(vmin, vmax):
            g["score"] = 0.5
        else:
            g["score"] = (v - vmin) / (vmax - vmin) if direction == "higher" else (vmax - v) / (vmax - vmin)
        return g
    df = df.groupby("data.video_id", group_keys=False).apply(norm_scores)

    # Aggregate by algorithm + settings
    key_cols = ["group", "exp_name", "module",
                "cfg.template_strategy" if "cfg.template_strategy" in df.columns else None,
                "cfg.gaussian_filtered" if "cfg.gaussian_filtered" in df.columns else None]
    key_cols = [k for k in key_cols if k is not None]

    agg = (df.groupby(key_cols, dropna=False)
             .agg(mean_score=("score","mean"),
                  mean_rank=("rank","mean"),
                  videos=("data.video_id","nunique"),
                  runs=("data.video_id","count"),
                  mean_runtime=("runtime_s","mean") if "runtime_s" in df.columns else ("rank","mean"),
                  m_ssim_mean=("m.ssim_mean","mean") if "m.ssim_mean" in df.columns else ("rank","mean"),
                  m_mse_mean=("m.mse_mean","mean") if "m.mse_mean" in df.columns else ("rank","mean"),
                  m_crispness=("m.crispness_improvement","mean") if "m.crispness_improvement" in df.columns else ("rank","mean"))
             .reset_index())

    # Order by mean_score desc, tie-breaker mean_rank asc
    agg = agg.sort_values(by=["mean_score","mean_rank"], ascending=[False, True])

    # Save leaderboard
    out_csv = runs_root / f"leaderboard_{metric.replace('.','_')}.csv"
    agg.to_csv(out_csv, index=False)

    # Print a tidy top-N and the absolute best row
    print("\n=== TOP configs (by normalized score) ===")
    show = [c for c in ["group","exp_name","cfg.template_strategy","cfg.gaussian_filtered",
                        "mean_score","mean_rank","videos","runs","mean_runtime",
                        "m_ssim_mean","m_mse_mean","m_crispness"]
            if c in agg.columns]
    print(agg[show].head(args.topn).to_string(index=False))

    best = agg.iloc[0].to_dict()
    print("\n=== BEST overall ===")
    for k in show:
        if k in best:
            print(f"{k}: {best[k]}")
    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()
