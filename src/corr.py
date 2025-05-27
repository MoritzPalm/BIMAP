from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from scipy.signal import correlate
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
from tqdm import tqdm
import ants
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error



pth = Path('./data/low_movement/Experiment-746czi')


def find_highest_correlation(frame_stack: np.array, plot=False):
    mean_corrs = []
    for i in range(frame_stack.shape[0]-1):
        corr_2d = correlate(frame_stack[i], frame_stack[i+1], method='auto')
        mean_corrs.append(np.mean(corr_2d))
    if plot:
        plt.plot(mean_corrs)
        plt.show()
    return int(np.argmax(mean_corrs))


def ants_reg(frame_stack: np.array, template_idx: int):
    motion_corrected_images = []
    fixed = ants.from_numpy(frame_stack[template_idx])
    
    for i in tqdm(range(frame_stack.shape[0])):
        moving = ants.from_numpy(frame_stack[i])
        areg = ants.registration(fixed, moving, 'SyN')
        motion_corrected_images.append(areg['warpedmovout'])

    return motion_corrected_images


def evaluate(corrected_images: list, template: np.array):
    ssim_list = []
    for i in tqdm(range(len(corrected_images))):
        ssim_list.append(ssim(template, corrected_images[i], 
                            data_range=template.max() - template.min()))
    gradient_ssim_list = []
    gX_template = sobel(template, axis=0)
    gY_template = sobel(template, axis=1)
    magnitude_template = np.sqrt((gX_template**2) + (gY_template**2))
    orientation_template = np.arctan2(gY_template, gX_template) * (180/np.pi) % 180
    data_range_template = magnitude_template.max() - magnitude_template.min()
    for i in range(len(corrected_images)):
        gX = sobel(corrected_images[i], axis=0)
        gY = sobel(corrected_images[i], axis=1)
        magnitude = np.sqrt((gX**2) + (gY**2))
        orientation = np.arctan2(gY, gX) * (180/np.pi) % 180
        gradient_ssim = ssim(magnitude, magnitude_template, data_range=data_range_template)
        gradient_ssim_list.append(gradient_ssim)
    return np.mean(ssim_list), np.mean(gradient_ssim_list)


def main():
    frame_paths = list(pth.glob('frame_*.tif'))
    frame_paths = [x.as_posix() for x in frame_paths]
    frame_paths.sort()
    frames = [np.array(Image.open(path)) for path in frame_paths]
    frames = np.asarray(frames)
    template_index = find_highest_correlation(frames, plot=False)
    print(f'{template_index=}')
    ants_corrected = ants_reg(frames, template_index)
    template = frames[template_index]
    mean_ssim = evaluate(ants_corrected, template)
    print(f'{mean_ssim}')


if __name__ == "__main__":
    main()
