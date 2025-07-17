import os
import numpy as np
import cv2
import ants

from bimap.src.utils import save_and_display_video
#from bimap.src.ants_test import template
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.models.core.model_utils import get_points_on_a_grid
import torch
import pandas as pd

import torch_lddmm
from utils import evaluate, find_highest_correlation
from cotracker_utils import warping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = "Experiment-591czi"


def get_metrics_for_single_experiment(path):
    video = read_video_from_path(path).squeeze()
    frames = []
    for i, frame in enumerate(video):
        # frame is shape (258, 512)
        resized_frame = cv2.resize(frame, (256, 256))
        frames.append(resized_frame)
    frames = frames#[:10]
    # opportunity
    #video = torch.from_numpy(video)[None].to(device).permute(0, 2, 1, 3,4).repeat(1,1,
    #                                                                             3,
    #                                                                             1,1).float()
    video = torch.from_numpy(np.expand_dims(video, axis=0)).float()
    video = torch.nn.functional.interpolate(video, size=(256,256),
                                            mode="bilinear", align_corners=False)[None]
    video = video.permute(0,2,1,3,4).repeat(1,1,3,1,1).to(device)#[:,:10,:,:,:]
    #template_idx = find_highest_correlation(frames)
    template_idx = 0
    ants_metrics = get_ants_metrics(frames, template_idx)
    #lddmm_metrics = get_lddmm_metrics(frames, template_idx)
    normcorre_metrics = get_normcorre_metrics(
        f'../../data/output/normcorre/{filename}_normcorre.tif',
                    frames[template_idx])
    cotracker_metrics = get_cotracker_metrics(video, frames, template_idx)

    print(f"ants: {ants_metrics}")
    #print(f"lddmms: {lddmm_metrics}")
    print(f"normcorre: {normcorre_metrics}")
    print(f"cotracker: {cotracker_metrics}")





def get_ants_metrics(frames, template_idx=0):
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

    ssims: list[float] = evaluate(results, frames[template_idx])

    return np.mean(ssims)


def get_lddmm_metrics(frames, template_idx=0):
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
    ssims: list[float] = evaluate(results, frames[template_idx])
    return np.mean(ssims)

def get_normcorre_metrics(results_path, template):
    # due to its install process issues, this function just imports precalculated
# results
    results = read_video_from_path(results_path).squeeze()
    ssims =  evaluate(results, template)
    return np.mean(ssims)

def get_cotracker_metrics(video, frames, template_idx=0, grid_size=15):
    model = torch.hub.load("facebookresearch/co-tracker",
                           "cotracker3_online").to(device)
    video.to(device)
    model(video_chunk=video, is_first_step=True, grid_size=grid_size)
    #model.model.model_resolution = video.shape[3:]
    #grid_pts = get_points_on_a_grid(
    #    grid_size, model.model.model_resolution)
    for idx in range(0, video.shape[1] - model.step, model.step):
        pred_tracks, pred_visibility = model(video_chunk=video[:, idx:idx +
                                                                    model.step * 2])
    result = warping(pred_tracks.cpu().numpy(),
                     np.expand_dims(frames, axis=-1))
    save_and_display_video(np.array(result), f'../../data/output/cotracker'
                                             f'/{filename}.mp4')
    ssims: list[float] = evaluate(result, frames[0])
    return np.mean(ssims)



if __name__ == '__main__':
    get_metrics_for_single_experiment(
        f"../../data/input/strong_movement/{filename}.tif")