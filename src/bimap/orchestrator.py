#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import multiprocessing as mp
import os
import socket
import sys
import traceback
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- utils ----------
def now_iso() -> str:
    return datetime.now(UTC).isoformat()

def load_yaml(path: str):
    import yaml  # pip install pyyaml
    with open(path) as f:
        return yaml.safe_load(f)

def cartesian(params: dict):
    if not params:
        yield {}
        return
    keys = list(params.keys())
    vals = [(v if isinstance(v, list) else [v]) for v in (params[k] for k in keys)]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo, strict=False))

def parse_module_map(s: str) -> dict:
    mapping = {}
    if not s: return mapping
    for pair in s.split(","):
        pair = pair.strip()
        if not pair: continue
        if "=" not in pair: raise ValueError(f"Bad module-map entry '{pair}' (want name=module)")
        k, v = pair.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping

def sanitize_slug(text: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))
    return s.strip("._") or "item"

# ---------- dataset handling ----------
def load_datasets_from_yaml(data: dict) -> list[dict]:
    vids = (((data or {}).get("datasets") or {}).get("videos")) or []
    if not isinstance(vids, list):
        raise ValueError("'datasets.videos' must be a list of {id, path[, category]}.")

    out = []
    for v in vids:
        if not isinstance(v, dict) or "path" not in v:
            raise ValueError("Each entry needs 'path' (and optional 'id', 'category').")
        vid_id = v.get("id") or Path(v["path"]).stem
        out.append({
            "id": str(vid_id),
            "path": str(v["path"]),
            "category": str(v.get("category", "") or ""),  # "" if unknown
        })
    if not out:
        raise ValueError("No videos found.")
    return out

def glob_videos(pattern: str) -> list[dict]:
    paths = sorted(Path(p).resolve() for p in map(str, Path().glob(pattern)) )
    # The above won't work for wildcards from CLI; do manual glob:
    import glob as _glob
    paths = sorted(Path(p).resolve() for p in _glob.glob(pattern))
    out = [{"id": Path(p).stem, "path": str(p)} for p in paths]
    if not out:
        raise ValueError(f"--data-glob matched 0 files: {pattern}")
    return out


def load_videos_manifest_csv(csv_path: str) -> list[dict]:
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("CSV must include 'path' (optional 'id','category').")
    out = []
    for _, r in df.iterrows():
        vid_id = (r["id"] if "id" in df.columns and pd.notna(r["id"]) else Path(r["path"]).stem)
        cat = (r["category"] if "category" in df.columns and pd.notna(r["category"]) else "")
        out.append({"id": str(vid_id), "path": str(r["path"]), "category": str(cat)})
    if not out:
        raise ValueError("CSV manifest is empty.")
    return out

def select_videos(all_videos: list[dict], selector) -> list[dict]:
    # selector can be: "*", ["id1","id2"], {ids:[...], categories:[...]}
    if selector in (None, "*", ["*"]):
        return all_videos

    if isinstance(selector, list):
        want_ids = set(map(str, selector))
        chosen = [v for v in all_videos if v["id"] in want_ids]
        missing = want_ids - {v["id"] for v in chosen}
        if missing:
            raise ValueError(f"Unknown video ids: {sorted(missing)}")
        return chosen

    if isinstance(selector, dict):
        ids = set(map(str, selector.get("ids", [])))
        cats = set(map(str, selector.get("categories", []))) if "categories" in selector else cats

        chosen = all_videos
        if cats:
            chosen = [v for v in chosen if v.get("category", "") in cats]
        if ids:
            chosen = [v for v in chosen if v["id"] in ids]
            missing = ids - {v["id"] for v in chosen}
            if missing:
                raise ValueError(f"Unknown video ids: {sorted(missing)}")
        if not chosen:
            raise ValueError("Selection matched 0 videos.")
        return chosen

    raise ValueError(f"Unsupported data selector: {selector!r}")


# ---------- schema expansion ----------
def normalize_groups(data: dict) -> dict:
    if "groups" not in data or not isinstance(data["groups"], dict):
        raise ValueError("Top-level 'groups' must be a mapping.")
    groups = {}
    for gname, g in data["groups"].items():
        g = g or {}
        exps = g.get("experiments", [])
        if exps is None: exps = []
        if not isinstance(exps, list): raise ValueError(f"'groups.{gname}.experiments' must be a list or [].")
        groups[gname] = {
            "module": g.get("module"),
            "defaults": g.get("defaults", {}) or {},
            "experiments": exps,
        }
    return groups

def expand_global_experiments(data: dict, group_names: list[str]) -> list[dict]:
    ge_list = data.get("global_experiments", []) or []
    out = []
    for ge in ge_list:
        name = ge.get("name")
        if not name: raise ValueError("Each global_experiment needs a 'name'.")
        apply_to = ge.get("apply_to", "*")
        if apply_to in ("*", ["*"]): targets = list(group_names)
        else:
            if not isinstance(apply_to, list): raise ValueError(f"'apply_to' of '{name}' must be list or '*'.")
            targets = apply_to
        out.append({
            "name": name,
            "params": ge.get("params", {}) or {},
            "per_group_overrides": ge.get("per_group_overrides", {}) or {},
            "data": ge.get("data", "*"),
            "targets": targets,
        })
    return out

def resolve_module_for_group(gname: str, ginfo: dict, module_map: dict) -> str:
    if ginfo.get("module"): return ginfo["module"]
    if gname in module_map: return module_map[gname]
    return f"{gname}"

def build_manifest(data: dict, module_map: dict, videos: list[dict]) -> list[dict]:
    groups = normalize_groups(data)
    global_exps = expand_global_experiments(data, list(groups.keys()))
    manifest = []

    for gname, g in groups.items():
        module = resolve_module_for_group(gname, g, module_map)
        g_default_params = deepcopy((g.get("defaults") or {}).get("params", {}))

        # Local experiments
        for exp in g.get("experiments", []):
            if "name" not in exp: raise ValueError(f"Experiment in group '{gname}' missing 'name'.")
            eparams = deepcopy(exp.get("params", {}))
            merged = {**g_default_params, **eparams}
            target_videos = select_videos(videos, exp.get("data", "*"))
            for cfg in cartesian(merged):
                for vid in target_videos:
                    manifest.append({
                        "group": gname, "exp_name": exp["name"], "module": module,
                        "config": cfg, "video": vid,
                    })

        # Global experiments
        for ge in global_exps:
            if gname not in ge["targets"]: continue
            merged = {**g_default_params, **(ge.get("params") or {})}
            over = (ge.get("per_group_overrides", {}) or {}).get(gname, {})
            merged = {**merged, **(over.get("params") or {})}
            target_videos = select_videos(videos, ge.get("data", "*"))
            for cfg in cartesian(merged):
                for vid in target_videos:
                    manifest.append({
                        "group": gname, "exp_name": ge["name"], "module": module,
                        "config": cfg, "video": vid,
                    })

    for m in manifest:
        m["run_id"] = str(uuid.uuid4())[:8]
    return manifest

# ---------- result normalization ----------
def compute_basic_summary(per_frame_df: pd.DataFrame) -> dict:
    out = {}
    for col in per_frame_df.columns:
        if col == "frame_idx": continue
        try:
            if np.issubdtype(per_frame_df[col].dtype, np.number):
                out[f"{col}_mean"] = float(per_frame_df[col].mean())
                out[f"{col}_std"]  = float(per_frame_df[col].std(ddof=0))
        except Exception:
            pass
    out["num_frames"] = len(per_frame_df)
    return out

def normalize_result_structure(res: dict):
    if not isinstance(res, dict):
        return pd.DataFrame({"frame_idx": []}), {}, {}, None
    runtime_s = res.get("runtime_s")
    artifacts = res.get("artifacts", {}) or {}
    if isinstance(res.get("metrics"), dict):
        pf = res["metrics"].get("per_frame") or {}
        per_frame_df = pd.DataFrame(pf) if pf else pd.DataFrame({"frame_idx": []})
        summary = res["metrics"].get("summary", {}) or {}
    else:
        pf = {k: v for k, v in res.items()
              if k not in {"runtime_s", "artifacts"} and isinstance(v, list)}
        if pf and "frame_idx" not in pf:
            first_key = next(iter(pf))
            pf["frame_idx"] = list(range(len(pf[first_key])))
        per_frame_df = pd.DataFrame(pf) if pf else pd.DataFrame({"frame_idx": []})
        summary = {}
    if "frame_idx" in per_frame_df.columns:
        try: per_frame_df["frame_idx"] = per_frame_df["frame_idx"].astype(int)
        except Exception: pass
    if not summary and len(per_frame_df) > 0:
        summary = compute_basic_summary(per_frame_df)
    return per_frame_df, summary, artifacts, runtime_s

# ---------- child process (sequential) ----------
def _child_worker(module_name: str, cfg: dict, run_dir: str, ret_path: str):
    run_dir_path = Path(run_dir)
    stdout_path = run_dir_path / "stdout.log"
    stderr_path = run_dir_path / "stderr.log"
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = open(stdout_path, "w")
    sys.stderr = open(stderr_path, "w")
    try:
        mod = importlib.import_module(module_name)
        fn = mod.run
        result = fn(cfg)
        payload = {"ok": True, "result": result}
    except Exception:
        payload = {"ok": False, "error": "exception", "traceback": traceback.format_exc()}
    finally:
        try:
            sys.stdout.close(); sys.stderr.close()
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    Path(ret_path).write_text(json.dumps(payload))

def run_once_sequential(module_name: str, run_cfg: dict, run_dir: Path, timeout_s: int = 0, retries: int = 0) -> dict:
    ret_file = run_dir / "_child_return.json"
    attempt = 0
    while True:
        attempt += 1
        if ret_file.exists(): ret_file.unlink()
        p = mp.Process(target=_child_worker, args=(module_name, run_cfg, str(run_dir), str(ret_file)))
        p.start()
        p.join(timeout=timeout_s if timeout_s > 0 else None)
        if p.is_alive():
            p.terminate(); p.join()
            res = {"ok": False, "error": "timeout"}
        else:
            res = json.loads(ret_file.read_text()) if ret_file.exists() else {"ok": False, "error": "child_crash_no_output"}
        if res.get("ok") or attempt > retries:
            return res

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Sequential orchestrator with dataset support")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--timeout-s", type=int, default=0)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--module-map", default="", help="ants=ants,cotracker=cotracker,...")
    ap.add_argument("--data-glob", default="", help="Override videos via glob, e.g. '/data/*.mp4'")
    ap.add_argument("--data-manifest", default="", help="CSV with columns path[,id] to override videos")
    args = ap.parse_args()

    data = load_yaml(args.config)
    module_map = parse_module_map(args.module_map)

    # Load videos (CLI overrides YAML)
    if args.data_glob:
        videos = glob_videos(args.data_glob)
    elif args.data_manifest:
        videos = load_videos_manifest_csv(args.data_manifest)
    else:
        videos = load_datasets_from_yaml(data)

    runs_root = Path(args.out); runs_root.mkdir(parents=True, exist_ok=True)
    table_path = runs_root / "table.parquet"

    manifest = build_manifest(data, module_map, videos)

    # Save manifest metadata
    meta = {
        "created_at": now_iso(),
        "host": socket.gethostname(),
        "git_commit": os.popen("git rev-parse --short HEAD").read().strip() or None,
        "count": len(manifest),
        "num_videos": len(videos),
    }
    (runs_root / f'{meta["created_at"].replace(":","-")}_manifest.json').write_text(json.dumps({"meta": meta, "runs": manifest}, indent=2))

    rows = []
    for i, m in enumerate(manifest, start=1):
        vid = m["video"]
        cat_slug = sanitize_slug(vid.get("category", "") or "uncat")
        vid_slug = sanitize_slug(vid["id"])
        exp_dir = runs_root / m["group"] / m["exp_name"] / cat_slug / vid_slug / f'run_{m["run_id"]}'
        artifacts_dir = exp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        run_cfg = dict(m["config"])
        run_cfg["data"] = {"id": vid["id"], "path": vid["path"], "category": vid.get("category", "")}
        run_cfg["run"] = {"run_dir": str(exp_dir), "artifacts_dir": str(artifacts_dir)}
        (exp_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))

        print(f"[{i}/{len(manifest)}] {m['group']}/{m['exp_name']} @ {vid['id']} -> {m['module']}  run_id={m['run_id']}")

        res = run_once_sequential(m["module"], run_cfg, exp_dir, timeout_s=args.timeout_s, retries=args.retries)

        row = {
            "group": m["group"],
            "exp_name": m["exp_name"],
            "module": m["module"],
            "run_id": m["run_id"],
            "created_at": now_iso(),
            "data.video_id": vid["id"],
            "data.path": vid["path"],
            "data.category": vid.get("category", ""),
            **{f"cfg.{k}": v for k, v in (m["config"] or {}).items()},
            "ok": res.get("ok", False),
            "error": res.get("error"),
            "duration_s": res.get("duration_s"),
        }

        if res.get("ok"):
            per_frame_df, summary, artifacts, runtime_s = normalize_result_structure(res["result"])
            pf_path = exp_dir / "per_frame.parquet"
            try:
                if len(per_frame_df): per_frame_df.to_parquet(pf_path, index=False)
                pf_rel = str(pf_path.relative_to(runs_root)) if pf_path.exists() else None
            except Exception:
                pf_path = exp_dir / "per_frame.csv"
                if len(per_frame_df): per_frame_df.to_csv(pf_path, index=False)
                pf_rel = str(pf_path.relative_to(runs_root)) if pf_path.exists() else None

            payload = {
                "runtime_s": runtime_s,
                "metrics": {"summary": summary},
                "artifacts": artifacts,
                "per_frame_path": pf_path.name if pf_path.exists() else None,
                "data": {"id": vid["id"], "path": vid["path"], "category": vid.get("category", "")},
            }
            (exp_dir / "result.json").write_text(json.dumps(payload, indent=2))

            row["runtime_s"] = runtime_s
            if pf_rel: row["per_frame_path"] = pf_rel
            for k, v in (summary or {}).items(): row[f"m.{k}"] = v
            for k, v in (artifacts or {}).items(): row[f"artifact.{k}"] = str(v)
        else:
            (exp_dir / "result.json").write_text(json.dumps(res, indent=2))

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        try:
            df.to_parquet(table_path, index=False)
            print(f"Global table: {table_path}")
        except Exception:
            csv_path = runs_root / "table.csv"
            df.to_csv(csv_path, index=False)
            print(f"No parquet engine; wrote CSV: {csv_path}")

    print(f"Completed {len(rows)} runs in {runs_root.resolve()}")
    print(f"Successes: {sum(r.get('ok') for r in rows)}, Failures: {sum(not r.get('ok') for r in rows)}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
