"""Module to perform motion correction on a video using the ANTs library."""

import time

import ants
import numpy as np
from utils import evaluate, find_highest_correlation, load_video, save_and_display_video
from floodfill import floodfill


def main() -> tuple[list[np.ndarray], dict, float]:
    """Use this function only for local testing purposes, you probably want to use run() instead."""
    path = "../../data/input/strong_movement/b5czi.tif"
    video, frames, filename = load_video(path, length=10, order="CTHW")
    template_idx = find_highest_correlation(frames)
    return _run(frames, template_idx, filename=filename, out_path="./output")

def run(config: dict) -> dict:
    """Run ANTs image registration on a video input.

    :param config: configuration dictionary with the following fields:
        data: path: path to the input video
        run:
            artifacts_dir: directory to save output artifacts
        method: (optional) method used for registration, default is "SyNOnly"
        template_strategy: (optional) strategy to select the template frame,
                            either "first" or "computed", default is "first"
        gaussian_filtered: (optional) whether to apply Gaussian filtering
    :return: dictionary with results and metrics
    """
    path = config["data"]["path"]
    output_path = config["run"]["artifacts_dir"]
    method = config.get("method")
    if method is None:
        method = "SyNOnly"

    filtered = config.get("gaussian_filtered", False)
    video, frames, filename = load_video(path, gaussian_filtered=filtered, length=400, order="CTHW")

    template_index = find_highest_correlation(frames) if config.get("template_strategy") == "computed" else 0
    warped, metrics, runtime = _run(frames, template_index, output_path, filename, method)
    warped = np.array(warped)
    floodfill(warped, output_path)
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


def _run(frame_stack: list[np.ndarray], template_idx: int,
         out_path: str, filename: str, ants_method: str = "SyNOnly") -> tuple[list[np.ndarray], dict, float]:
    """Run internal ANTs image registration on a stack of frames.

    :param frame_stack: list of 2D numpy arrays representing the video frames
    :param template_idx: index of the template frame in the frame_stack
    :param out_path: path to save the output video
    :param filename: name of the output video file (without extension)
    :param ants_method: method used for registration, default is "SyNOnly"
    :return: tuple of (motion_corrected_images, metrics, runtime)
        motion_corrected_images: list of motion corrected frames
        metrics: dictionary with evaluation metrics
        runtime: time taken to perform the registration
    """
    motion_corrected_images = []
    fixed = ants.from_numpy(frame_stack[template_idx])
    start_time = time.time()
    for i in range(len(frame_stack)):
        moving = ants.from_numpy(frame_stack[i])
        areg = ants.registration(fixed, moving, ants_method)
        motion_corrected_images.append(areg["warpedmovout"].numpy().astype(np.float32))
    end_time = time.time()
    save_and_display_video(np.array(motion_corrected_images), f"{out_path}/{filename}.tif")
    metrics = evaluate(motion_corrected_images, frame_stack, frame_stack[template_idx])
    return motion_corrected_images, metrics, end_time - start_time


if __name__ == "__main__":
    main()
