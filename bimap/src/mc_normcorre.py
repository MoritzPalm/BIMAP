import subprocess
from pathlib import Path
import os

import numpy as np

from utils import load_video, evaluate, find_highest_correlation

def run_in_caiman(env_name, work_dir, script_path, *args):
    activate_bat = r"C:\Users\morit\Miniforge3\Scripts\activate.bat"
    cmd = (
        f'cd /d "{work_dir}" '
        f'&& CALL "{activate_bat}" "{env_name}" '
        f'&& python "{script_path}" {" ".join(args)}'
    )
    result = subprocess.run(
        cmd,
        shell=True,
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
    stdout, runtime = run_in_caiman("caiman", r"C:\Users\morit\caiman_data\demos\notebooks", "mc_normcorre.py", abs_path, abs_output_path)
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
    out = run_in_caiman("caiman", r"C:\Users\morit\caiman_data\demos\notebooks",
                        "mc_normcorre.py", "C:/Users/morit/Documents/Studium/BIMAP/data/input/strong_movement/b5czi.tif", "C:/Users/morit/Documents/Studium/BIMAP/data/output/normcorre")
    print(out)