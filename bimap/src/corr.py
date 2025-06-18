"""utils for bimap image registration."""

from pathlib import Path

import ants
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from scipy.signal import correlate
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import cv2

#pth = Path("../../data/low_movement/Experiment-746czi")
pth =  Path("../../data/strong_movement/Experiment-591czi")

def load_example_experiment() -> list[np.array]:
    """Load Experiment-746czi."""
    pattern = r"frame_*.tif"
    frame_paths = list(pth.glob(pattern))
    if not frame_paths:
        error_msg = f"No files found matching {pattern=}"
        raise FileNotFoundError(error_msg)
    return [handle_raw_image(path) for path in frame_paths]


def handle_raw_image(path: Path) -> np.ndarray:
    image = Image.open(path.as_posix())
    arr = np.array(image).byteswap().newbyteorder()
    arr_f32 = arr.astype(np.float32) / 65535.0
    return arr_f32


def get_magnitude(img: np.array) -> np.array:
    """Calculate the magnitude of the gradient of the image using sobel filters."""
    g_x = sobel(img, axis=0)
    g_y = sobel(img, axis=1)
    return np.sqrt((g_x**2) + (g_y**2))


def find_highest_correlation(frame_stack: list[np.array], *, plot: bool =False) -> int:
    """Find frame with maximum correlation to previous frame."""
    corrs = []
    for i in range(len(frame_stack)-1):
        corr = np.corrcoef(frame_stack[i].flatten(), frame_stack[i+1].flatten())[0,1]
        corrs.append(corr)
    max_idx = int(np.argmax(corrs))
    if plot:
        plt.plot(max_idx, corrs[max_idx], "x")
        plt.plot(corrs)
        plt.title("Correlation of each frame with the previous")
        plt.show()
    return max_idx


def ants_reg(frame_stack: list[np.array], template_idx: int) -> list[np.array]:
    """Image Registration using the AnTsPy package."""
    motion_corrected_images = []
    fixed = ants.from_numpy(frame_stack[template_idx])

    for i in tqdm(range(len(frame_stack))):
        moving = ants.from_numpy(frame_stack[i])
        areg = ants.registration(fixed, moving, "SyN")
        motion_corrected_images.append(areg["warpedmovout"].numpy().astype(np.float32))

    return motion_corrected_images


def evaluate(corrected_images: list[np.array], template: np.array) -> tuple[list]:
    """Evaluate the image registration based on the SSIM of the gradient image."""
    ssim_list = [ssim(template, moving, data_range=template.max() - template.min()) for moving in corrected_images]
    gradient_ssim_list = []
    magnitude_template = get_magnitude(template)
    #data_range_template = magnitude_template.max() - magnitude_template.min()
    #for i in range(len(corrected_images)):
        #magnitude = get_magnitude(corrected_images[i])
        #gradient_ssim = ssim(magnitude, magnitude_template, data_range=data_range_template)
        #gradient_ssim_list.append(gradient_ssim)
    return ssim_list


def float32_to_uint8(image: np.array) -> np.array:
    """Convert float23 image type to uint8 image type."""
    min_val, max_val = image.min(), image.max()
    return ((image - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)


def uint8_to_float32(image: np.array) -> np.array:
    """Convert uint8 image type to float23 image type."""
    image = image.astype(np.float32)
    return image / 255.0


def save_results(corrected_images: list[np.array], path: Path, method: str) -> None:
    """Save the results of the image registration."""
    save_path = path / (method + "_results")
    Path.mkdir(save_path, parents=True, exist_ok=True)
    for i, img in enumerate(corrected_images):
        image = Image.fromarray(img)
        filename = save_path / f"corrected_{i}.tif"
        image.save(filename)

def denoise_stack(imgs: list[np.array]) -> np.array:
    imgs8 = [float32_to_uint8(img) for img in imgs]
    return [cv2.bilateralFilter(img8, d=10, sigmaColor=20, sigmaSpace=50) for img8 in imgs8]

def normalize_stack(imgs: list[np.array]) -> np.array:
    pass

