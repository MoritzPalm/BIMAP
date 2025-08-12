"""utils for bimap image registration."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter
from skimage.metrics import structural_similarity as ssim
import torch
from cotracker.utils.visualizer import Visualizer, read_video_from_path

#pth = Path("../../data/low_movement/Experiment-746czi")
pth =  Path("../../data/input/strong_movement/Experiment-591czi")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_example_experiment() -> list[np.ndarray]:
    """Load Experiment-746czi."""
    pattern = r"frame_*.tif"
    frame_paths = list(pth.glob(pattern))
    if not frame_paths:
        error_msg = f"No files found matching {pattern=}"
        raise FileNotFoundError(error_msg)
    return [handle_raw_image(path) for path in frame_paths]


def handle_raw_image(path: Path) -> np.ndarray:
    """Load image from path and convert from <u2 to float32 format."""
    image = Image.open(path.as_posix())
    arr = np.array(image).byteswap().newbyteorder()
    arr_f32 = arr.astype(np.float32) / 65535.0
    return arr_f32


def get_magnitude(img: np.array) -> np.array:
    """Calculate the magnitude of the gradient of the image using sobel filters."""
    g_x = sobel(img, axis=0)
    g_y = sobel(img, axis=1)
    return np.sqrt((g_x**2) + (g_y**2))


def find_highest_correlation(frame_stack: list[np.ndarray], *, plot: bool=False) -> int:
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


def evaluate(corrected_images: np.array, images, template: np.ndarray) -> dict:
    """Evaluate the image registration based on the SSIM of the gradient image."""
    ssim_list = [float(ssim(template, moving, data_range=template.max() - template.min())) for moving in corrected_images]
    mse_list = [float(quantized_mse(template, moving)) for moving in corrected_images]
    summary_image_before = np.mean(images, axis=0)
    summary_image_after = np.mean(corrected_images, axis=0)
    crispness_before = crispness(summary_image_before)
    crispness_after = crispness(summary_image_after)
    crispness_improvement = float(crispness_after - crispness_before)
    results = {"ssims": ssim_list, "mse_list": mse_list, "crispness_improvement": crispness_improvement}
    return results


def float32_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float23 image type to uint8 image type."""
    min_val, max_val = image.min(), image.max()
    return ((image - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)


def uint8_to_float32(image: np.ndarray) -> np.ndarray:
    """Convert uint8 image type to float23 image type."""
    image = image.astype(np.float32)
    return image / 255.0


def save_results(corrected_images: list[np.ndarray], path: Path, method: str) -> None:
    """Save the results of the image registration."""
    save_path = path / (method + "_results")
    Path.mkdir(save_path, parents=True, exist_ok=True)
    for i, img in enumerate(corrected_images):
        image = Image.fromarray(img)
        filename = save_path / f"corrected_{i}.tif"
        image.save(filename)

def denoise_video(video: torch.Tensor, sigma=1) -> torch.Tensor:
    filtered = np.empty_like(video)
    for t in range(video.shape[0]):
        filtered[t] = gaussian_filter(video[t], sigma=sigma)
    return filtered


def denoise_stack(imgs: list[np.ndarray], sigma=1) -> list[np.ndarray]:
    for i in range(len(imgs)):
         imgs[i] = gaussian_filter(imgs[i], sigma=sigma)
    return imgs


def save_and_display_video(array, filename='output.mp4', fps=30):
    num_frames, height, width = array.shape

    # Normalize and convert to uint8 if needed
    if array.dtype != np.uint8:
        array_min = array.min()
        array_max = array.max()
        array = 255 * (array - array_min) / (array_max - array_min + 1e-8)
        array = np.clip(array, 0, 255).astype(np.uint8)

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)

    for i in range(num_frames):
        frame = array[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # <- This line is critical
        out.write(frame)

    out.release()
    print(f"Video saved to {filename}")

def quantize_image(image, method='percentile', thresholds=None):
    """
    Quantize a calcium imaging frame into 3 levels:
    0 - background, 1 - base, 2 - high activity

    params:
        image (np.ndarray): 2D image
        method (str): 'percentile' or 'manual'
        thresholds (tuple): if method='manual', provide (low_thresh, high_thresh)

    Returns:
        np.ndarray: quantized image with values 0, 1, 2
    """
    if method == 'percentile':
        low_thresh = np.percentile(image, 10)
        high_thresh = np.percentile(image, 90)
    elif method == 'manual' and thresholds:
        low_thresh, high_thresh = thresholds
    else:
        raise ValueError("Invalid method or missing thresholds")

    quantized = np.zeros_like(image, dtype=np.uint8)
    quantized[(image >= low_thresh) & (image < high_thresh)] = 1
    quantized[image >= high_thresh] = 2

    return quantized

def quantized_mse(image1, image2, method='percentile', thresholds=None):
    """
    Compute MSE between two quantized calcium imaging frames.

    params:
        image1 (np.ndarray): reference frame
        image2 (np.ndarray): registered frame
        method (str): quantization method
        thresholds (tuple): manual threshold values if used

    Returns:
        float: mean squared error between quantized images
    """
    q1 = quantize_image(image1, method=method, thresholds=thresholds)
    q2 = quantize_image(image2, method=method, thresholds=thresholds)
    mse = np.mean((q1 - q2) ** 2)
    return mse


def crispness(image):
    """
    calculates the crispness value, intended to be used on a summary image before and after registration.
    """
    gradient = get_magnitude(image)
    abs_gradient = np.abs(gradient)
    norm = np.linalg.norm(abs_gradient, ord="fro")
    return norm


def load_video(path, len=-1, gaussian_filtered=False):
    filename = Path(path).stem
    video = read_video_from_path(path)
    if video is None:
        raise FileNotFoundError(f"Video in {path} not found")
    video = video.squeeze()
    if gaussian_filtered:
        filtered = np.empty_like(video)
        for t in range(video.shape[0]):
            filtered[t] = gaussian_filter(video[t], sigma=2)
        video = filtered
    frames = np.array([frame for frame in video], dtype=np.float32)
    video = torch.from_numpy(np.expand_dims(video.astype(np.float32), axis=0)).float()
    if len != -1:
        frames = frames[:len]
    video = torch.from_numpy(np.expand_dims(video, axis=0)).float()
    video = video.permute(0, 2, 1, 3, 4).repeat(1, 1, 3, 1, 1).to(device)[:,:len,:,:,:]
    return video, frames, filename

def get_all_paths(input_folder) -> list:
    p = Path(input_folder)
    paths = list(p.rglob("*.tif"))
    file_paths = [p for p in paths if p.is_file()]
    return file_paths

