import math

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import ants
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

pth = Path('./data/low_movement/Experiment-746czi')


def main():
    all_frames = list(pth.glob('frame_*.tif'))
    all_frames = [x.as_posix() for x in all_frames]
    all_frames.sort()

    fixed = ants.image_read(all_frames[0])
    motion_corrected = []
    ssim_errors = []
    mse_errors = []
    for i in range(len(all_frames)):
        moving = ants.image_read(all_frames[i])
        areg = ants.registration(fixed, moving, 'SyN')
        motion_corrected.append(areg['warpedmovout'])
        ssim_errors.append(ssim(fixed.numpy(), moving.numpy(), data_range=moving.numpy().max() - moving.numpy().min()))
        mse_errors.append(mean_squared_error(fixed.numpy(), moving.numpy()))
    mean_ssim = sum(ssim_errors)/len(ssim_errors)
    mean_mse = sum(mse_errors)/len(ssim_errors)
    print(f"mean SSIM: {mean_ssim}")
    print(f"mean MSE: {mean_mse}")
    print(f"RMSE: {math.sqrt(mean_mse)}")


if __name__ == "__main__":
    main()
