import subprocess
from pathlib import Path
import os
import shlex

import numpy as np

from utils import load_video, evaluate, find_highest_correlation


def _find_conda_sh():
    """
    Return the path to conda's activation script: <base>/etc/profile.d/conda.sh
    Tries env vars, `conda info --base`, then common install paths.
    """
    # 1) From CONDA_EXE (e.g., /home/you/miniforge3/condabin/conda)
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        base = Path(conda_exe).parent.parent  # .../condabin -> base
        candidate = base / "etc" / "profile.d" / "conda.sh"
        if candidate.exists():
            return str(candidate)

    # 2) Ask conda directly (works if 'conda' is on PATH)
    try:
        out = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True, text=True, check=True
        )
        base = Path(out.stdout.strip())
        candidate = base / "etc" / "profile.d" / "conda.sh"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass

    # 3) Common locations
    for base in [
        Path.home() / "miniforge3",
        Path.home() / "mambaforge",
        Path("/opt/miniforge3"),
        Path("/opt/mambaforge"),
        Path("/usr/local/miniforge3"),
    ]:
        candidate = base / "etc" / "profile.d" / "conda.sh"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not locate conda.sh. Try running `conda info --base` in a terminal "
        "to find your Miniforge/Mambaforge base, then use <base>/etc/profile.d/conda.sh."
    )

def run_in_caiman(env_name, work_dir, script_path, *args):
    conda_sh = _find_conda_sh()
    q = shlex.quote
    quoted_args = " ".join(q(str(a)) for a in args)

    cmd = (
        f'set -e\n'
        f'source {q(conda_sh)}\n'
        f'conda activate {q(env_name)}\n'
        f'cd {q(work_dir)}\n'
        f'python {q(script_path)} {quoted_args}\n'
    )

    result = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Error:\n{result.stderr}")

    runtime = None
    for line in result.stdout.splitlines():
        if line.startswith("TIME_TAKEN="):
            runtime = float(line.split("=")[1])
            break
    return result.stdout, runtime

def run(config:dict) -> dict:
    path = config["data"]["path"]
    abs_path = os.path.abspath(path)
    video, frames, filename = load_video(path)
    output_path = config["run"]["artifacts_dir"]
    abs_output_path = os.path.abspath(output_path)
    filtered = config.get("gaussian_filtered", False)
    template_index = config.get("template_strategy", None)
    if config.get("template_strategy", None) == "computed":
        template_index = find_highest_correlation(frames)
    else:
        template_index = 0
    stdout, runtime = run_in_caiman("caiman", r"/data/ih26ykel/caiman_data/demos/notebooks", "mc_normcorre.py", abs_path, abs_output_path)
    warped, _, _ = load_video(f"{output_path}/{filename}.tif")
    metrics = evaluate(warped.cpu().numpy().squeeze()[:,0,:,:], frames, frames[template_index])
    ssim_list = metrics["ssims"]
    mse_list = metrics["mse_list"]
    crispness_improvement = metrics["crispness_improvement"]
    metrics = {
        "per_frame": {
            "ssim": ssim_list,
            "mse": mse_list
        },
        "summary": {
            "mse_mean": float(np.mean(mse_list)),
            "mse_std": float(np.std(mse_list)),
            "crispness_improvement": crispness_improvement
        }
    }
    result = {"runtime_s": runtime,
              "metrics": metrics,
              "artifacts": {
                  "output_path": f"{output_path}/{filename}.mp4",
              }}
    return result


if __name__ == '__main__':
    out = run_in_caiman("caiman", r"/data/ih26ykel/caiman_data/demos/notebooks",
                        "mc_normcorre.py", "/data/ih26ykel/BIMAP/data/input/strong_movement/b5czi.tif", "/data/ih26ykel/BIMAP/data/output/normcorre")
    print(out)