import time

import numpy as np
import ants

from utils import load_video, save_and_display_video, find_highest_correlation, \
    evaluate, denoise_stack,  denoise_video


def main():
    """
    this function is only used for local testing purposes,
    you probably want to use the run() function
    """
    path = "../../data/input/strong_movement/b5czi.tif"
    video, frames, filename = load_video(path, len=10)
    #template_idx = find_highest_correlation(frames)
    template_idx = 0
    result = _run(frames, template_idx, filename=filename)


def run(config: dict):
    """
    main entrypoint to run the ANTs image registration
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
    method = config.get("method", None)
    if method is None:
        method = "SyNOnly"

    filtered = config.get("gaussian_filtered", False)
    video, frames, filename = load_video(path, gaussian_filtered=filtered)

    if config.get("template_strategy", None) == "computed":
        template_index = find_highest_correlation(frames)
    else:
        template_index = 0
    warped, metrics, runtime = _run(frames, template_index, output_path, filename, method)
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


def _run(frame_stack: list[np.ndarray], template_idx: int, out_path: str, filename: str, ants_method: str = "SyNOnly"):
    """Image Registration using the AnTsPy package."""
    motion_corrected_images = []
    fixed = ants.from_numpy(frame_stack[template_idx])
    start_time = time.time()
    for i in range(len(frame_stack)):
        moving = ants.from_numpy(frame_stack[i])
        areg = ants.registration(fixed, moving, ants_method)
        motion_corrected_images.append(areg["warpedmovout"].numpy().astype(np.float32))
    end_time = time.time()
    save_and_display_video(np.array(motion_corrected_images), f'{out_path}/{filename}.mp4')
    metrics = evaluate(motion_corrected_images, frame_stack, frame_stack[template_idx])
    return motion_corrected_images, metrics, end_time - start_time


if __name__ == "__main__":
    main()