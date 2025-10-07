"""Module for performing LDDMM-based motion correction on video frames."""

import itertools
import time
from typing import TypedDict, Unpack

import numpy as np
import torch_lddmm
from utils import evaluate, find_highest_correlation, load_video, save_and_display_video
from floodfill import floodfill


class LDDMMParams(TypedDict, total=False):
    """Changeable Parameters for LDDMM registration."""

    a: int
    epsilon: float
    nt: int
    niter: int
    sigma: float
    sigmaR: float


def main() -> dict:
    """Use this function only for local testing purposes, you probably want to use run() instead."""
    path = "../../data/input/strong_movement/b5czi.tif"
    video, frames, filename = load_video(path, length=10, order="CTHW")
    #frames = frames.astype(np.int16)
    config = {
        "run": {
            "artifacts_dir": "./artifacts", },
        "data": {
            "path": path, }
    }
    return run(config)


def run(config:dict) -> dict:
    """Run LDDMM motion correction on a video input.

    :param config: configuration dictionary with the following fields:
                    data: path: path to the input video
                    run:
                        artifacts_dir: directory to save output artifacts
                    template_strategy: (optional) strategy to select the template frame,
                                        either "first" or "computed", default is "first"
                    gaussian_filtered: (optional) whether to apply Gaussian filtering
    :return: dictionary with results and metrics
    """
    path = config["data"]["path"]
    output_path = config["run"]["artifacts_dir"]
    filtered = config.get("gaussian_filtered", False)
    video, frames, filename = load_video(path, gaussian_filtered=filtered, length=400, order="CTHW")
    template_index = find_highest_correlation(frames) if config.get("template_strategy") == "computed" else 0
    param_dict = {"a": 8, "epsilon": 1.0, "nt": 7,
                  "niter": 200, "sigma": 10, "sigmaR": 10}
    warped, metrics, runtime = _run(frames, template_index, output_path, filename,
                                    save=True, **param_dict)
    warped = np.array(warped)
    #floodfill(warped, output_path)
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


def lddmms_grid_search(frames: list[np.ndarray], filename: str, template_idx: int) \
        -> list[tuple[np.ndarray, dict]]:
    """Perform a grid search over LDDMM parameters.

    :param frames: list of video frames as numpy arrays
    :param filename: name of the input video file
    :param template_idx: index of the template frame
    :return: list of tuples containing the results and the parameters used
    for each combination of parameters
    """
    results = []
    min_param = 1
    max_param = 10
    step_size = 1
    parameter_list = ["a", "epsilon", "sigma", "sigmaR"]
    values = list(range(min_param, max_param + 1, step_size))
    combinations = list(itertools.product(values, repeat=len(parameter_list)))
    for combo in combinations:
        params = dict(zip(parameter_list, combo, strict=False))
        result = _run(frames, template_idx, "./output", filename, save=False, **params)
        results.append((result, params))
    return results


def _run(frames: list, template_idx: int, output_path: str,
         filename: str,*, save: bool, **kwargs: Unpack[LDDMMParams]) \
        -> tuple[list[np.ndarray], dict, float]:
    target = frames[template_idx]
    results = []
    start_time = time.time()
    for i in range(1, len(frames)-1):
        lddmm = torch_lddmm.LDDMM(template=frames[i],
                                  target=target,
                                  do_affine=0,
                                  do_lddmm=1,
                                  **kwargs,
                                  optimizer="gdr",
                                  dx=[1.0,1.0],
                                  gpu_number=0,
                                  verbose=False)
        _ = lddmm.run()
        results.append(lddmm.outputDeformedTemplate()[0])
    end_time = time.time()
    if save:
        save_and_display_video(np.array(results), f"{output_path}/{filename}.tif")
    metrics = evaluate(results, frames, target)
    return results, metrics, end_time - start_time


if __name__ == "__main__":
    main()
