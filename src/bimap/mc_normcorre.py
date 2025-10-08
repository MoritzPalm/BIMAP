"""Run CaImAn's Normcorre motion correction algorithm via a subprocess call."""

import logging
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
from utils import evaluate, find_highest_correlation, load_video, save_and_display_video
from floodfill import floodfill

logger = logging.getLogger(__name__)

# WARNING: this way of using normcorre is littered with security issues, use at your own risk!


def _find_conda_sh() -> str:
    """Find the conda.sh script to enable conda environments in bash.

    :return: the path to conda's activation script: <base>/etc/profile.d/conda.sh

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
        # this is potentially a security issue if conda is not trusted as arbitrary code
        # could be executed if the PATh variable is manipulated
        out = subprocess.run(
            ["conda", "info", "--base"], #noqa: S607
            capture_output=True, text=True, check=True,
        )
        base = Path(out.stdout.strip())
        candidate = base / "etc" / "profile.d" / "conda.sh"
        if candidate.exists():
            return str(candidate)
    except FileNotFoundError as e:
        logger.debug("conda not found on PATH: %s", e)
    except subprocess.CalledProcessError as e:
        logger.debug("conda command failed: %s", e)

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

    msg = (
        "Could not locate conda.sh. "
        "Try running `conda info --base` in a terminal to find your "
        "Miniforge/Mambaforge base, then use <base>/etc/profile.d/conda.sh."
    )
    raise FileNotFoundError(msg)


def run_in_caiman(env_name: str, work_dir: str, script_path: str, *args: str) \
        -> tuple[str, float|None]:
    """Run a script in a conda environment via a subprocess call.

    :param env_name: name of the conda environment to activate
    :param work_dir: working directory to change to before running the script
    :param script_path: path to the python script to run
    :param args: arguments to pass to the script
    :return: tuple of (stdout, runtime) where stdout is the captured standard output
             containing the resulting metrics
             and runtime is the time taken as reported by the script (or None if not found)
    """
    conda_sh = _find_conda_sh()
    q = shlex.quote
    quoted_args = " ".join(q(str(a)) for a in args)
    os.environ["CAIMAN_TEMP"] = "/data/ih26ykel/caiman_data/temp"

    cmd = (
        f"set -e\n"
        f"source {q(conda_sh)}\n"
        f"conda activate {q(env_name)}\n"
        f"cd {q(work_dir)}\n"
        f"python {q(script_path)} {quoted_args}\n"
    )

    result = subprocess.run(    # noqa: S602
        cmd,
        check=False, shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        msg = f"Subprocess failed with exit code {result.returncode}\nSTDERR:\n{stderr}"
        raise RuntimeError(msg)

    runtime = None
    for line in result.stdout.splitlines():
        if line.startswith("TIME_TAKEN="):
            runtime = float(line.split("=")[1])
            break
    return result.stdout, runtime


def run(config:dict) -> dict:
    """Run CaImAn's Normcorre motion correction on a video input.

    :param config: configuration dictionary with the following fields:
                    data: path: path to the input video
                    run:
                          artifacts_dir: directory to save output artifacts
                    template_strategy: (optional) strategy to select the template frame,
                                        either "first" or "computed", default is "first"
                    gaussian_filtered: (optional) whether to apply Gaussian filtering
    :return: dictionary with results and metrics
    """
    path = Path(config["data"]["path"])
    abs_path = Path.resolve(path)
    filtered = config.get("gaussian_filtered", False)
    video, frames, filename = load_video(str(abs_path), length=400, order="CTHW", gaussian_filtered=filtered)
    video = np.squeeze(video[:, :, 0, :, :])
    save_and_display_video(video, "temp_input.tif")
    output_path = Path(config["run"]["artifacts_dir"])
    abs_output_path = Path.resolve(output_path)
    print(abs_output_path)
    input_path = Path("temp_input.tif").resolve()
    template_index = find_highest_correlation(frames) if config.get("template_strategy") == "computed" else 0
    stdout, runtime = run_in_caiman("caiman",
                                    r"/data/ih26ykel/caiman_data/demos/notebooks",
                                    "mc_normcorre_callee.py",
                                    str(input_path),
                                    str(abs_output_path),
                                    filename)
    warped, _, _ = load_video(f"{output_path}/artifacts/{filename}.tif", gaussian_filtered=False, length=400, order="CTHW")
    #floodfill(warped, output_path)
    metrics = evaluate(np.squeeze(warped[:,:,0,:,:]), frames, frames[template_index])
    ssim_list = metrics["ssims"]
    mse_list = metrics["mse_list"]
    crispness_improvement = metrics["crispness_improvement"]
    metrics = {
        "per_frame": {
            "ssim": ssim_list,
            "mse": mse_list,
        },
        "summary": {
            "mse_mean": float(np.mean(mse_list)),
            "mse_std": float(np.std(mse_list)),
            "crispness_improvement": crispness_improvement,
        },
    }
    return {"runtime_s": runtime,
              "metrics": metrics,
              "artifacts": {
                  "output_path": f"{output_path}/{filename}.tif",
              }}


if __name__ == "__main__":
    config = {
        "data": {
            "path": "/data/ih26ykel/BIMAP/data/input/strong_movement/b5czi.tif",
        },
        "run": {
            "artifacts_dir": "/data/ih26ykel/BIMAP/data/output/normcorre",
        }
    }
    result = run(config)
    #out = run_in_caiman("caiman",
    #                    r"/data/ih26ykel/caiman_data/demos/notebooks",
    #                    "mc_normcorre_callee.py",
    #                    "/data/ih26ykel/BIMAP/data/input/strong_movement/b5czi.tif",
    #                    "/data/ih26ykel/BIMAP/data/output/normcorre")
    #logger.debug(out)
    print(result)
    logger.debug(result)
