import time
import itertools

import numpy as np
import torch_lddmm

from utils import load_video, save_and_display_video, find_highest_correlation, evaluate, denoise_video, denoise_stack

def main():
    path = "../../data/input/strong_movement/b5czi.tif"
    video, frames, filename = load_video(path)
    frames = frames.astype(np.int16)
    #template_idx = find_highest_correlation(frames)
    template_idx = 0
    results = _run(frames, template_idx, "../../data/output/lddmms", filename, save=True)
    return results


def run(config:dict) -> dict:
    path = config["data"]["path"]
    output_path = config["run"]["artifacts_dir"]
    filtered = config.get("gaussian_filtered", False)
    video, frames, filename = load_video(path, gaussian_filtered=filtered)
    if config.get("template_strategy", None) == "computed":
        template_index = find_highest_correlation(frames)
    else:
        template_index = 0
    param_dict = {"a": 8, "epsilon": 1.0, "nt": 7,
                  "niter": 200, "sigma": 10, "sigmaR": 10}
    warped, metrics, runtime = _run(frames, template_index, output_path, filename, True,
                                    **param_dict)
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

def lddmms_grid_search(frames, filename, template_idx):
    results = []
    min_param = 1
    max_param = 10
    step_size = 1
    parameter_list = ["a", "epsilon", "sigma", "sigmaR"]
    values = list(range(min_param, max_param + 1, step_size))
    combinations = list(itertools.product(values, repeat=len(parameter_list)))
    for combo in combinations:
        params = dict(zip(parameter_list, combo, strict=False))
        result = _run(frames, filename, template_idx, save=False, **params)
        results.append((result, params))
    return results


def _run(frames, template_idx, output_path, filename, save: bool, **kwargs):
    target = frames[template_idx]
    results = []
    start_time = time.time()
    for i in range(1, len(frames)-1):
        lddmm = torch_lddmm.LDDMM(template=frames[i],
                                  target=target,
                                  do_affine=0,
                                  do_lddmm=1,
                                  **kwargs,
                                  optimizer='adam',
                                  dx=[1.0,1.0],
                                  gpu_number=0,
                                  verbose=False)
        _ = lddmm.run()
        results.append(lddmm.outputDeformedTemplate()[0])
    end_time = time.time()
    if save:
        save_and_display_video(np.array(results), f'{output_path}/{filename}.mp4')
    metrics = evaluate(results, frames, target)
    return results, metrics, end_time - start_time


if __name__ == "__main__":
    main()