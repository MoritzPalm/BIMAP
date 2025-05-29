from pathlib import Path

import ants
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import sobel
from scipy.signal import correlate
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

pth = Path("../data/low_movement/Experiment-746czi")

def get_magnitude(img: np.array) -> np.array:
    """Calculate the magnitude of the gradient of the image using sobel filters."""
    g_x = sobel(img, axis=0)
    g_y = sobel(img, axis=1)
    return np.sqrt((g_x**2) + (g_y**2))


def find_highest_correlation(frame_stack: np.array, *, plot: bool =False) -> int:
    """Find frame with maximum correlation to previous frame."""
    mean_corrs = []
    for i in range(frame_stack.shape[0]-1):
        corr_2d = correlate(frame_stack[i], frame_stack[i+1], method="auto")
        mean_corrs.append(np.mean(corr_2d))
    if plot:
        plt.plot(mean_corrs)
        plt.title("Correlation of each frame with the previous")
        plt.show()
    return int(np.argmax(mean_corrs))


def ants_reg(frame_stack: np.array, template_idx: int) -> list[np.array]:
    """Image Registration using the AnTsPy package."""
    motion_corrected_images = []
    fixed = ants.from_numpy(frame_stack[template_idx])

    for i in tqdm(range(frame_stack.shape[0])):
        moving = ants.from_numpy(frame_stack[i])
        areg = ants.registration(fixed, moving, "SyN")
        motion_corrected_images.append(areg["warpedmovout"].numpy().astype(np.float32))

    return motion_corrected_images


def evaluate(corrected_images: list, template: np.array) -> tuple[list, list]:
    """Evaluate the image registration based on the SSIM of the gradient image."""
    ssim_list = []
    for i in tqdm(range(len(corrected_images))):
        ssim_list.append(ssim(template, corrected_images[i],
                            data_range=template.max() - template.min()))
    gradient_ssim_list = []
    magnitude_template = get_magnitude(template)
    data_range_template = magnitude_template.max() - magnitude_template.min()
    for i in range(len(corrected_images)):
        magnitude = get_magnitude(corrected_images[i])
        gradient_ssim = ssim(magnitude, magnitude_template, data_range=data_range_template)
        gradient_ssim_list.append(gradient_ssim)
    return ssim_list, gradient_ssim_list

