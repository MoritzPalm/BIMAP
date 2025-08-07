import time
import itertools

import numpy as np
import torch_lddmm

from utils import load_video, save_and_display_video, find_highest_correlation, evaluate

def main():
    path = "../../data/input/strong_movement/b5czi.tif"
    video, frames, filename = load_video(path)
    #template_idx = find_highest_correlation(frames)
    template_idx = 0
    results = run_lddmms(frames, filename, template_idx, save=True)


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
        result = run_lddmms(frames, filename, template_idx, save=False, **params)
        results.append((result, params))
    return results


def run_lddmms(frames, filename, template_idx, save: bool, **kwargs):
    target = frames[template_idx]
    results = []
    start_time = time.time()
    for i in range(1, len(frames)-1):
        lddmm = torch_lddmm.LDDMM(template=frames[i],
                                  target=target,
                                  do_affine=0,
                                  do_lddmm=1,
                                  **kwargs,
                                  optimizer='gdr',
                                  gpu_number=0,
                                  verbose=False)
        _ = lddmm.run()
        results.append(lddmm.outputDeformedTemplate()[0])
    end_time = time.time()
    if save:
        save_and_display_video(np.array(results), f"../../data/output/lddmm/{filename}.mp4")
    metrics = evaluate(results, frames, target)
    return results, metrics, end_time - start_time


if __name__ == "__main__":
    main()