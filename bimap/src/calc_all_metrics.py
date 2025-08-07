import os
import numpy as np
import cv2
import ants
from pathlib import Path, PurePath
import time

from utils import save_and_display_video
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.models.core.model_utils import get_points_on_a_grid
import torch
import pandas as pd

import torch_lddmm
from utils import evaluate, find_highest_correlation
from mc_cotracker import warping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = "Experiment-746czi"
cotracker_grid_size = 15


def get_metrics_for_single_experiment(path):
    filename = Path(path).stem
    video = read_video_from_path(path).squeeze()
    frames = []
    for i, frame in enumerate(video):
        # frame is shape (258, 512)
        resized_frame = cv2.resize(frame, (256, 256))
        frames.append(resized_frame)
    frames = frames#[:10]
    #video = torch.from_numpy(video)[None].to(device).permute(0, 2, 1, 3,4).repeat(1,1,
    #                                                                             3,
    #                                                                             1,1).float()
    video = torch.from_numpy(np.expand_dims(video, axis=0)).float()
    video = torch.nn.functional.interpolate(video, size=(256,256),
                                            mode="bilinear", align_corners=False)[None]
    video = video.permute(0,2,1,3,4).repeat(1,1,3,1,1).to(device)#[:,:3,:,:,:]
    #template_idx = find_highest_correlation(frames)
    template_idx = 0
    print("calculating ants metrics")
    start_time = time.time()
    ants_metrics = get_ants_metrics(frames, filename, template_idx)
    ants_time = time.time() - start_time
    print("calculating lddmm metrics")
    start_time = time.time()
    lddmm_metrics = get_lddmm_metrics(frames, filename, template_idx)
    lddmm_time = time.time() - start_time
    print("calculating normcorre metrics")
    #normcorre_metrics = get_normcorre_metrics(
    #    f'../../data/output/normcorre/{filename}_normcorre.tif', frames,
    #                frames[template_idx])
    print("calculating cotracker metrics")
    start_time = time.time()
    cotracker_metrics = get_cotracker_metrics(video, frames, filename, template_idx,
                                              cotracker_grid_size)
    cotracker_time = time.time() - start_time
    print(f"ants: {ants_metrics}")
    print(f"lddmms: {lddmm_metrics}")
    #print(f"normcorre: {normcorre_metrics}")
    print(f"cotracker: {cotracker_metrics}")
    results = {"ants": ants_metrics,
               "lddms": lddmm_metrics,
               #"normcorre": normcorre_metrics,
               "cotracker": cotracker_metrics}

    times = {"ants": ants_time,
             "lddms": lddmm_time,
             #"normcorre": normcorre_time,
             "cotracker": cotracker_time
             }

    return results, times


def get_ants_metrics(frames, filename, template_idx=0):
    results = []
    for i in range(len(frames) - 1):
        fixed_image = ants.from_numpy(frames[template_idx])  # .transpose(1,0)
        moving_image = ants.from_numpy(frames[i + 1])  # .transpose(1,0)
        # Perform registration
        registration_result = ants.registration(fixed=fixed_image, moving=moving_image,
                                                type_of_transform="SyNOnly")  # ,  reg_iterations = [100,1000,20])
        # Transform the moving image
        warped_image = registration_result['warpedmovout'].numpy()
        results.append(np.array(warped_image))
    save_and_display_video(np.array(results),
                           f'../../data/output/ants/{filename}.mp4')

    ssims, mse, crispness = evaluate(results, frames, frames[template_idx])

    return np.mean(ssims), np.mean(mse), crispness


def get_lddmm_metrics(frames, filename, template_idx=0):
    results = []
    for i in range(1, len(frames) - 1):
        lddmm = torch_lddmm.LDDMM(template=[frames[template_idx]],
                                  target=[frames[i]],
                                  # outdir='../notebook/',
                                  do_affine=0,
                                  do_lddmm=1,
                                  a=7,
                                  nt=5,
                                  niter=200,
                                  epsilon=4e0,
                                  sigma=20.0,
                                  sigmaR=40.0,
                                  optimizer='gdr',
                                  gpu_number=0,
                                  verbose=False,
                                  dx=None)

        _ = lddmm.run()
        results.append(lddmm.outputDeformedTemplate()[0])
    ssims, mse, crispness = evaluate(results, frames, frames[template_idx])
    return np.mean(ssims), np.mean(mse), crispness


def get_normcorre_metrics(results_path, images,template):
    # due to its install process issues, this function just imports precalculated
# results
    results = read_video_from_path(results_path).squeeze()
    ssims, mse, crispness =  evaluate(results, images, template)
    return np.mean(ssims), np.mean(mse), crispness


def get_cotracker_metrics(video, frames, filename, template_idx=0, grid_size=15):
    model = torch.hub.load("facebookresearch/co-tracker",
                           "cotracker3_offline").to(device)
    video.to(device)
    model.model.model_resolution = video.shape[3:]
    pred_tracks, pred_visibility = model(video=video,
                                         grid_size=grid_size,
                                         grid_query_frame=template_idx,
                                         backward_tracking=True)

    grid_pts = get_points_on_a_grid(grid_size, model.model.model_resolution)

    result = warping(pred_tracks.cpu().numpy(),
                     np.expand_dims(frames, axis=-1))
    save_and_display_video(np.array(result), f'../../data/output/cotracker'
                                             f'/{filename}.mp4')
    ssims, mse, crispness = evaluate(result, frames, frames[0])
    return np.mean(ssims), np.mean(mse), crispness


def get_voxelmorph_metrics(results_path, images,template):
    pass


def get_all_results(path_list):
    results = []
    for path in path_list:
        results.append(get_metrics_for_single_experiment(path))


def get_all_paths(input_folder) -> list:
    p = Path(input_folder)
    paths = list(p.rglob("*.tif"))
    file_paths = [p for p in paths if p.is_file()]
    return file_paths



if __name__ == '__main__':
    #get_metrics_for_single_experiment(
    #    f"../../data/input/low_movement/{filename}.tif")
    paths = get_all_paths(f"../../data/input")
    results = get_all_results(paths)