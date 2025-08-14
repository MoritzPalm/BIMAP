#!/usr/bin/env python3
"""
analyze_runs.py — Schema-aware analysis for the current orchestrator.

Understands the orchestrator's columns:
  - group, exp_name, module, run_id, created_at
  - data.video_id, data.category, per_frame_path, runtime_s, ok, error
  - cfg.* (config params), m.* (summary metrics), artifact.*
and per-run files:
  - <run_dir>/per_frame.parquet (or per_frame.csv)
  - <run_dir>/result.json

Outputs under runs/reports/:
  Global (mode=all):
    - all/leaderboard_<metric>.csv
    - all/best_by_algo_<metric>.csv
    - all/best_by_category_<metric>.csv (if categories exist)
    - all/best_overall_<metric>.json
    - all/per_video_ranks_<metric>.csv (optional)
    - plots: topN bar + per-group boxplots
  Per-video (mode=video):
    - <video_id>[_<category>]/summary_per_experiment.csv
    - <video_id>[_<category>]/leaderboard_<metric>.csv
    - <video_id>[_<category>]/best_overview.json
    - plots: per-frame line plots + mean±std bars

Usage examples:
  python analyze_runs.py --runs-root runs --mode all --metric auto
  python analyze_runs.py --runs-root runs --mode video --video-id VID123 --metric m.ssim_mean
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- orchestrator-aware I/O ----------
def read_runs_table(root: Path) -> pd.DataFrame:
    p_parq = root / "table.parquet"
    p_csv  = root / "table.csv"
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Expected {p_parq} or {p_csv}")

def run_dir_from_row(root: Path, row: pd.Series) -> Path | None:
    """Prefer per_frame_path; fallback to searching derived path."""
    if "per_frame_path" in row and isinstance(row["per_frame_path"], str) and row["per_frame_path"]:
        return (root / Path(row["per_frame_path"]).parent).resolve()
    # Fallback (layout: runs/<group>/<exp>/<cat>/<video>/run_<id>/)
    group = row.get("group"); exp = row.get("exp_name"); vid = row.get("data.video_id")
    cat = row.get("data.category") or "uncat"
    if group and exp and vid:
        base = root / str(group) / str(exp) / str(cat) / str(vid)
        cand = list(base.glob(f"run_{row.get('run_id','*')}"))
        if cand: return cand[0].resolve()
    return None

def load_per_frame(run_dir: Path) -> pd.DataFrame:
    pf = run_dir / "per_frame.parquet"
    if pf.exists(): return pd.read_parquet(pf)
    csv = run_dir / "per_frame.csv"
    if csv.exists(): return pd.read_csv(csv)
    return pd.DataFrame({"frame_idx": []})

def load_result_json(run_dir: Path) -> dict:
    p = run_dir / "result.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}

# ---------- metrics & ranking ----------
LOWER_IS_BETTER_HINTS = ("loss","mse","mae","rmse","error","epe","ae","nrmse")
HIGHER_IS_BETTER_HINTS = ("ssim","ncc","psnr","dice","iou","accuracy","f1","auc","crispness","mi")

def available_metrics(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("m.")])

def guess_primary_metric(metrics: list[str]) -> str | None:
    prefs = ["m.ssim_mean","m.ncc_mean","m.psnr_mean","m.dice_mean","m.mse_mean","m.rmse_mean"]
    for p in prefs:
        if p in metrics: return p
    return metrics[0] if metrics else None

def metric_direction(metric: str) -> str:
    m = metric.lower()
    if any(h in m for h in LOWER_IS_BETTER_HINTS): return "lower"
    if any(h in m for h in HIGHER_IS_BETTER_HINTS): return "higher"
    return "higher"

def compute_summaries_from_per_frame(df_pf: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for c in df_pf.columns:
        if c == "frame_idx" or not pd.api.types.is_numeric_dtype(df_pf[c]):
            continue
        v = df_pf[c].astype(float)
        out[f"m.{c}_mean"] = float(v.mean())
        out[f"m.{c}_std"]  = float(v.std(ddof=0))
    out["num_frames"] = int(len(df_pf))
    return out

def ensure_summary_row(root: Path, row: pd.Series) -> pd.Series:
    """
    Make sure row has m.* summaries. If missing, load per_frame and compute,
    returning an updated row (does not mutate disk table).
    """
    has_any_m = any(str(col).startswith("m.") and not pd.isna(row[col]) for col in row.index if col in row)
    if has_any_m:
        return row
    rdir = run_dir_from_row(root, row)
    if not rdir or not rdir.exists():
        return row
    pf = load_per_frame(rdir)
    if len(pf) == 0:
        return row
    sums = compute_summaries_from_per_frame(pf)
    for k, v in sums.items():
        row[k] = v
    # also patch runtime if missing
    if pd.isna(row.get("runtime_s", np.nan)):
        resj = load_result_json(rdir)
        rs = resj.get("runtime_s") if isinstance(resj, dict) else None
        if rs is not None:
            row["runtime_s"] = float(rs)
    return row

def rank_per_video(df: pd.DataFrame, metric: str, direction: str) -> pd.DataFrame:
    use = df.copy()
    if "ok" in use.columns: use = use[use["ok"] == True]
    use = use[pd.notna(use[metric])]
    use[metric] = pd.to_numeric(use[metric], errors="coerce")
    use = use[pd.notna(use[metric])]
    if use.empty: return use
    ascending = (direction == "lower")
    use["rank"] = use.groupby("data.video_id")[metric].rank(method="average", ascending=ascending)
    def norm(g):
        v = g[metric].to_numpy(dtype=float)
        if v.size == 0:
            g["score"] = np.nan
            return g
        vmin, vmax = float(np.min(v)), float(np.max(v))
        if np.isclose(vmin, vmax):
            g["score"] = 0.5
        else:
            g["score"] = (v - vmin) / (vmax - vmin) if direction=="higher" else (vmax - v) / (vmax - vmin)
        return g
    return use.groupby("data.video_id", group_keys=False).apply(norm)

def aggregate_configs(df_ranked: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["group","exp_name","module"] + [c for c in df_ranked.columns if c.startswith("cfg.")]
    agg = (df_ranked
           .groupby(key_cols, dropna=False)
           .agg(mean_rank=("rank","mean"),
                std_rank=("rank","std"),
                mean_score=("score","mean"),
                videos=("data.video_id","nunique"),
                runs=("data.video_id","count"),
                mean_runtime=("runtime_s","mean") if "runtime_s" in df_ranked.columns else ("rank","mean"))
           .reset_index())
    agg["order_key"] = (-agg["mean_score"].fillna(-1e9), agg["mean_rank"].fillna(1e9))
    return agg.sort_values("order_key").drop(columns=["order_key"])

def best_per_group(agg: pd.DataFrame) -> pd.DataFrame:
    idx = agg.groupby("group")["mean_score"].idxmax()
    return agg.loc[idx].sort_values("mean_score", ascending=False)

def best_per_category(df_ranked: pd.DataFrame) -> pd.DataFrame:
    if "data.category" not in df_ranked.columns:
        return pd.DataFrame()
    key_cols = ["data.category","group","exp_name","module"] + [c for c in df_ranked.columns if c.startswith("cfg.")]
    agg = (df_ranked
           .groupby(key_cols, dropna=False)
           .agg(mean_score=("score","mean"),
                videos=("data.video_id","nunique"),
                runs=("data.video_id","count"))
           .reset_index())
    idx = agg.groupby(["data.category"])["mean_score"].idxmax()
    return agg.loc[idx].sort_values(["mean_score"], ascending=False)

# ---------- plotting ----------
def plot_topN_bar(agg: pd.DataFrame, metric: str, out_png: Path, topn: int = 10):
    df = agg.head(topn).copy()
    labels = [f"{g}/{e}" for g,e in zip(df["group"], df["exp_name"])]
    x = np.arange(len(df))
    fig = plt.figure(figsize=(10,5.5)); ax = fig.add_subplot(111)
    ax.bar(x, df["mean_score"].to_numpy())
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"mean_score (normalized from {metric})")
    ax.set_title(f"Top {topn} configs")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_box_by_group(df_ranked: pd.DataFrame, col: str, out_png: Path, title: str):
    df = df_ranked.copy()
    order = df.groupby("group")[col].median().sort_values(ascending=False).index.tolist()
    data = [df[df["group"]==g][col].dropna().to_numpy() for g in order]
    fig = plt.figure(figsize=(10,5.5)); ax = fig.add_subplot(111)
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel(col); ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_per_frame_lines(per_frame_list: List[dict], metric_name: str, out_png: Path):
    fig = plt.figure(figsize=(10,5.5)); ax = fig.add_subplot(111)
    ax.set_title(f"{metric_name} over frames")
    for item in per_frame_list:
        df_pf = item["pf"]
        if metric_name not in df_pf.columns:
            continue
        x = df_pf["frame_idx"].values if "frame_idx" in df_pf.columns else np.arange(len(df_pf[metric_name]))
        y = df_pf[metric_name].values
        label = f"{item['group']}/{item['exp_name']} ({item['run_id'][-4:]})"
        ax.plot(x, y, label=label)
    ax.set_xlabel("frame"); ax.set_ylabel(metric_name); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

# ---------- pipeline ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--mode", choices=["all","video"], default="all", help="analyze all videos or a single video")
    ap.add_argument("--video-id", default="", help="[mode=video] which data.video_id to analyze")
    ap.add_argument("--category", default="", help="[mode=video] optional data.category filter")
    ap.add_argument("--filter-group", default="", help="[all] only include this algorithm (group)")
    ap.add_argument("--filter-category", default="", help="[all] only include this dataset category")
    ap.add_argument("--metric", default="auto", help="primary metric (e.g., m.ssim_mean) or 'auto'")
    ap.add_argument("--topn", type=int, default=10, help="Top-N for bar chart")
    ap.add_argument("--dump-per-video", action="store_true", help="[all] save per-video ranks CSV")
    ap.add_argument("--plot-metrics", default="", help="[video] comma list of per-frame metrics to plot")
    args = ap.parse_args()

    root = Path(args.runs_root)
    df = read_runs_table(root)

    # Ensure m.* present using orchestrator per_frame outputs (only for rows missing them)
    patched_rows = []
    for _, r in df.iterrows():
        if not any(str(c).startswith("m.") and not pd.isna(r[c]) for c in df.columns):
            r = ensure_summary_row(root, r)
        patched_rows.append(r)
    df = pd.DataFrame(patched_rows)

    # choose metric
    mets = available_metrics(df)
    if args.metric == "auto":
        metric = guess_primary_metric(mets)
        if not metric:
            raise SystemExit("No summary metrics (m.*) available after patching.")
        print(f"[info] auto-selected primary metric: {metric}")
    else:
        metric = args.metric
        if metric not in df.columns:
            raise SystemExit(f"Metric '{metric}' not found. Available: {', '.join(mets)}")
    direction = metric_direction(metric)
    print(f"[info] optimizing for '{metric}' (direction={direction})")

    if args.mode == "all":
        # optional filters
        if args.filter_group:
            df = df[df["group"].astype(str) == str(args.filter_group)]
        if args.filter_category and "data.category" in df.columns:
            df = df[df["data.category"].astype(str) == str(args.filter_category)]
        if df.empty:
            raise SystemExit("No runs after filtering.")

        needed = ["group","exp_name","module","run_id","data.video_id","ok","runtime_s",metric] + \
                 [c for c in df.columns if c.startswith("cfg.")] + \
                 (["data.category"] if "data.category" in df.columns else [])
        needed = [c for c in needed if c in df.columns]
        ranked = rank_per_video(df[needed], metric, direction)
        if ranked.empty:
            raise SystemExit("No valid rows for ranking.")

        agg = aggregate_configs(ranked)
        reports = root / "reports" / "all"
        reports.mkdir(parents=True, exist_ok=True)
        metr_slug = metric.replace(".","_")

        # save artifacts
        (reports / f"leaderboard_{metr_slug}.csv").write_text(agg.to_csv(index=False))
        per_group = best_per_group(agg)
        per_group.to_csv(reports / f"best_by_algo_{metr_slug}.csv", index=False)
        per_cat = best_per_category(ranked)
        if not per_cat.empty:
            per_cat.to_csv(reports / f"best_by_category_{metr_slug}.csv", index=False)
        (reports / f"best_overall_{metr_slug}.json").write_text(json.dumps({
            "metric": metric, "direction": direction, "best_overall": agg.iloc[0].to_dict()
        }, indent=2))
        if args.dump_per_video:
            ranked.to_csv(reports / f"per_video_ranks_{metr_slug}.csv", index=False)

        # plots
        plot_topN_bar(agg, metric, reports / f"top{args.topn}_bar__{metr_slug}.png", topn=args.topn)
        plot_box_by_group(ranked, "score", reports / f"box__score_by_group__{metr_slug}.png",
                          f"Per-video normalized scores by group (from {metric})")
        plot_box_by_group(ranked, metric, reports / f"box__metric_by_group__{metr_slug}.png",
                          f"Per-video raw {metric} by group")

        # console summary
        show_cols = ["group","exp_name","module","videos","mean_score","mean_rank","mean_runtime"]
        show_cols += [c for c in agg.columns if c.startswith("cfg.")][:5]
        print("\n=== TOP 10 OVERALL ===")
        print(agg[show_cols].head(10).to_string(index=False))
        if not per_group.empty:
            print("\n=== BEST PER GROUP ===")
            print(per_group[show_cols].to_string(index=False))
        if not per_cat.empty:
            print("\n=== BEST PER CATEGORY ===")
            print(per_cat.to_string(index=False))
        print("\nSaved reports to:", reports)

    else:  # mode == "video"
        if not args.video_id:
            raise SystemExit("--video-id is required for mode=video")
        sel = (df["data.video_id"].astype(str) == str(args.video_id))
        if args.category and "data.category" in df.columns:
            sel &= (df["data.category"].astype(str) == str(args.category))
        dfv = df[sel].copy()
        if dfv.empty:
            raise SystemExit("No runs found for the specified video/category.")

        # compute leaderboard using table summaries (patched if needed)
        # also build per-frame collection using per_frame_path for chosen plots
        per_frame_items = []
        rows = []
        for _, r in dfv.iterrows():
            rdir = run_dir_from_row(root, r)
            pf = load_per_frame(rdir) if rdir else pd.DataFrame({"frame_idx":[]})
            per_frame_items.append({"group": r["group"], "exp_name": r["exp_name"], "run_id": r["run_id"], "pf": pf})
            drow = {
                "group": r["group"], "exp_name": r["exp_name"], "run_id": r["run_id"],
                "runtime_s": float(r["runtime_s"]) if "runtime_s" in r and not pd.isna(r["runtime_s"]) else np.nan,
            }
            for c in df.columns:
                if str(c).startswith("m."):
                    drow[c] = r[c]
            rows.append(drow)
        summary_df = pd.DataFrame(rows)

        # ensure metric exists
        if metric not in summary_df.columns:
            raise SystemExit(f"Primary metric '{metric}' not present for this video after patching.")

        # leaderboard (higher score means better; tie-breaker runtime)
        direction = metric_direction(metric)
        summary_df["score"] = summary_df[metric] if direction=="higher" else -summary_df[metric]
        leaderboard = summary_df.sort_values(["score","runtime_s"], ascending=[False, True]).reset_index(drop=True)

        tag = args.video_id + (f"_{args.category}" if args.category else "")
        out_dir = root / "reports" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        metr_slug = metric.replace(".","_")
        summary_df.to_csv(out_dir / "summary_per_experiment.csv", index=False)
        leaderboard.to_csv(out_dir / f"leaderboard_{metr_slug}.csv", index=False)

        # plots
        if args.plot_metrics:
            plot_list = [m.strip() for m in args.plot_metrics.split(",") if m.strip()]
        else:
            # auto-pick up to 3 numeric per-frame cols from first run
            cols = [c for c in per_frame_items[0]["pf"].columns if c != "frame_idx"]
            plot_list = cols[:3]
        for mname in plot_list:
            plot_per_frame_lines(per_frame_items, mname, out_dir / f"per_frame__{mname}.png")

        # best overview
        best = leaderboard.iloc[0].to_dict()
        (out_dir / "best_overview.json").write_text(json.dumps({
            "video_id": args.video_id,
            "category": args.category or None,
            "metric": metric,
            "direction": direction,
            "best": best
        }, indent=2))

        print("\n=== BEST FOR VIDEO {} ===".format(args.video_id))
        show_cols = ["group","exp_name","run_id","runtime_s",metric]
        print(leaderboard[show_cols].head(1).to_string(index=False))
        print("\nSaved reports to:", out_dir)


if __name__ == "__main__":
    main()
