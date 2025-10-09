"""utils for bimap image registration."""

from pathlib import Path
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from cotracker.utils.visualizer import read_video_from_path
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage.util import img_as_float32
import tifffile

pth =  Path("../../data/input/strong_movement/Experiment-591czi")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_example_experiment() -> list[np.ndarray]:
    """Load example experiment using the global example path.

    :return: list of 2D numpy arrays representing the video frames
    """
    pattern = r"frame_*.tif"
    frame_paths = list(pth.glob(pattern))
    if not frame_paths:
        error_msg = f"No files found matching {pattern=}"
        raise FileNotFoundError(error_msg)
    return [handle_raw_image(path) for path in frame_paths]


def handle_raw_image(path: Path) -> np.ndarray:
    """Load image from path and convert from <u2 to float32 format.

    :param path: path to the image file
    :return: image as a float32 numpy array with values in [0, 1]
    """
    image = Image.open(path.as_posix())
    arr = np.array(image).byteswap().newbyteorder()
    return arr.astype(np.float32) / 65535.0


def get_magnitude(img: np.array) -> np.array:
    """Calculate the magnitude of the gradient of the image using sobel filters.

    :param img: 2D numpy array representing the image
    :return: 2D numpy array representing the magnitude of the gradient
    """
    g_x = sobel(img, axis=0)
    g_y = sobel(img, axis=1)
    return np.sqrt((g_x**2) + (g_y**2))


def find_highest_correlation(frame_stack: list[np.ndarray], *, plot: bool=False) -> int:
    """Find frame with maximum correlation to previous frame.

    :param frame_stack: list of 2D numpy arrays representing the video frames
    :param plot: whether to plot the correlation values
    :return: index of the frame with the highest correlation to the previous frame
    """
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


def evaluate(corrected_images: np.array, images: list[np.array], template: np.ndarray) -> dict:
    """Evaluate the image registration based on the SSIM of the gradient image.

    :param corrected_images: list of 2D numpy arrays representing the motion corrected frames
    :param images: list of 2D numpy arrays representing the original video frames
    :param template: 2D numpy array representing the template frame
    :return: dictionary with evaluation metrics
    """
    ssim_list = [float(ssim(template, moving,
                            data_range=template.max() - template.min()))
                 for moving in corrected_images]
    mse_list = [float(quantized_mse(template, moving)) for moving in corrected_images]
    summary_image_before = np.mean(images, axis=0)
    summary_image_after = np.mean(corrected_images, axis=0)
    crispness_before = crispness(summary_image_before)
    crispness_after = crispness(summary_image_after)
    crispness_improvement = float(crispness_after - crispness_before)
    crispness_pct_improvement = crispness_improvement / crispness_before * 100
    corrs_list = []
    for corrected_image in corrected_images:
        corrs_list.append(np.corrcoef(corrected_image.flatten(), summary_image_after.flatten())[0,1])
    return {"ssims": ssim_list,
               "mse_list": mse_list,
               "corrs_list": corrs_list,
               "crispness_before": crispness_before,
               "crispness_after": crispness_after,
               "crispness_improvement": crispness_improvement,
               "crispness_pct_improvement": crispness_pct_improvement,
               }


def float32_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float23 image type to uint8 image type.

    :param image: input image as a float32 numpy array
    :return: image as a uint8 numpy array
    """
    min_val, max_val = image.min(), image.max()
    return ((image - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)


def uint8_to_float32(image: np.ndarray) -> np.ndarray:
    """Convert uint8 image type to float23 image type.

    :param image: input image as a uint8 numpy array
    :return: image as a float32 numpy array
    """
    image = image.astype(np.float32)
    return image / 255.0


def save_results(corrected_images: list[np.ndarray], path: Path, method: str) -> None:
    """Save the results of the image registration.

    :param corrected_images: list of 2D numpy arrays representing the motion corrected frames
    :param path: path to the directory where the results will be saved
    :param method: name of the registration method used
    """
    save_path = path / (method + "_results")
    Path.mkdir(save_path, parents=True, exist_ok=True)
    for i, img in enumerate(corrected_images):
        image = Image.fromarray(img)
        filename = save_path / f"corrected_{i}.tif"
        image.save(filename)

def denoise_video(video: torch.Tensor, sigma: int=1) -> torch.Tensor:
    """Apply Gaussian filter to each frame in the video tensor.

    :param video: 4D torch tensor representing the video (T, H, W, C)
    :param sigma: standard deviation for Gaussian kernel
    :return: 4D torch tensor representing the denoised video
    """
    filtered = np.empty_like(video)
    for t in range(video.shape[0]):
        filtered[t] = gaussian_filter(video[t], sigma=sigma)
    return filtered


def denoise_stack(imgs: list[np.ndarray], sigma: int=1) -> list[np.ndarray]:
    """Apply Gaussian filter to each image in the stack.

    :param imgs: list of 2D numpy arrays representing the video frames
    :param sigma: standard deviation for Gaussian kernel
    :return: list of 2D numpy arrays representing the denoised frames
    """
    for i in range(len(imgs)):
         imgs[i] = gaussian_filter(imgs[i], sigma=sigma)
    return imgs


def save_and_display_video(array: np.ndarray, filename: str="output.mp4", fps: int=30) -> None:
    """Save a numpy array as a video file using OpenCV.

    :param array (np.ndarray): 3D numpy array of shape (num_frames, height, width)
    :param filename (str): output video file name
    :param fps (int): frames per second for the output video
    """
    num_frames, height, width = array.shape

    # Normalize and convert to uint8 if needed
    if array.dtype != np.uint8:
        array_min = array.min()
        array_max = array.max()
        array = 255 * (array - array_min) / (array_max - array_min + 1e-8)
        array = np.clip(array, 0, 255).astype(np.uint8)

    ext = os.path.splitext(filename)[1].lower()

    if ext in (".tif", ".tiff"):
        # Save as a multipage TIFF stack (grayscale)
        # Note: fps doesn't apply to TIFF; no per-frame color conversion needed
        tifffile.imwrite(filename, array, photometric="minisblack")
        return

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)

    for i in range(num_frames):
        frame = array[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # <- This line is critical
        out.write(frame)

    out.release()


def quantize_image(image: np.ndarray, method: str="percentile",
                   thresholds: tuple[float, float]|None=None) -> np.ndarray:
    """Quantize a calcium imaging frame into 3 levels: 0 - background, 1 - base, 2 - high activity.

    :param image (np.ndarray): 2D image
    :param method (str): 'percentile' or 'manual'
    :param thresholds (tuple): if method='manual', provide (low_thresh, high_thresh)

    :return: np.ndarray: quantized image with values 0, 1, 2

    """
    if method == "percentile":
        low_thresh = np.percentile(image, 10)
        high_thresh = np.percentile(image, 90)
    elif method == "manual" and thresholds:
        low_thresh, high_thresh = thresholds
    else:
        raise ValueError("Invalid method or missing thresholds")

    quantized = np.zeros_like(image, dtype=np.uint8)
    quantized[(image >= low_thresh) & (image < high_thresh)] = 1
    quantized[image >= high_thresh] = 2

    return quantized

def quantized_mse(image1: np.ndarray, image2: np.ndarray, method: str="percentile",
                  thresholds: tuple[float, float]|None=None) -> float:
    """Compute MSE between two quantized calcium imaging frames.

    :param image1 (np.ndarray): reference frame
    :param image2 (np.ndarray): registered frame
    :param method (str): quantization method
    :param thresholds (tuple): manual threshold values if used

    :return (float): mean squared error between quantized images

    """
    q1 = quantize_image(image1, method=method, thresholds=thresholds)
    q2 = quantize_image(image2, method=method, thresholds=thresholds)
    return np.mean((q1 - q2) ** 2)


def crispness(image: np.ndarray) -> float:
    """Calculate the crispness value, intended to be used on a summary image before and after registration.

    :param image: 2D numpy array representing the image
    :return: float representing the crispness value
    """
    gradient = get_magnitude(image)
    abs_gradient = np.abs(gradient)
    return np.linalg.norm(abs_gradient, ord="fro")


def load_video(
    path: str,
    length: int = 400,
    gaussian_filtered: bool = False,
    out_dtype: np.dtype = np.float32,
    order: str = "auto",   # "auto", "THWC", "TCHW", "CTHW", "HWC", "CHW", "THW", "HW"
    full_channels: bool = True
) -> tuple[np.ndarray, list, str]:
    """
    Load a video from path, allow various input shape orders, convert to grayscale, and
    return (T, H, W) as float in [0,1] (by default float32).

    Parameters
    ----------
    order : str
        - "auto": infer layout heuristically.
        - Explicit options:
          "THWC", "TCHW", "CTHW", "HWC", "CHW", "THW" (gray), "HW" (gray single frame).

    Returns
    -------
    gray_thw : np.ndarray
        (B, T, C, H, W), dtype=out_dtype, values in [0,1].
    filename : str
        File stem.
    """
    filename = Path(path).stem
    arr = read_video_from_path(path)
    if arr is None:
        raise FileNotFoundError(f"Video in {path} not found")
    arr = np.asarray(arr)

    def to_thwc(x: np.ndarray, order: str) -> np.ndarray:
        """Normalize x into (T, H, W, C) (C in {1,3,4})."""
        if order != "auto":
            if order == "THWC":
                y = x
            elif order == "TCHW":
                y = np.transpose(x, (0, 2, 3, 1))
            elif order == "CTHW":
                y = np.transpose(x, (1, 2, 3, 0))
            elif order == "HWC":
                y = x[None, ...]
            elif order == "CHW":
                y = np.transpose(x, (1, 2, 0))[None, ...]
            elif order == "THW":  # grayscale with time
                y = x[..., None]
            elif order == "HW":   # single grayscale frame
                y = x[None, ..., None]
            else:
                raise ValueError(f"Unsupported order='{order}'")
            return y

        # --- auto detection heuristics ---
        nd = x.ndim
        if nd == 4:
            T, A, B, C = x.shape
            # If last dim looks like channels
            if C in (1, 3, 4):
                return x  # THWC
            # If second dim looks like channels: TCHW
            if A in (1, 3, 4):
                return np.transpose(x, (0, 2, 3, 1))
            # If first dim looks like channels: CTHW
            if T in (1, 3, 4):
                return np.transpose(x, (1, 2, 3, 0))
            raise ValueError(f"Cannot infer order for shape {x.shape}; please set order explicitly.")
        elif nd == 3:
            A, B, C = x.shape
            # HWC if last dim looks like channels
            if C in (1, 3, 4):
                return x[None, ...]  # -> THWC
            # CHW if first dim looks like channels
            if A in (1, 3, 4):
                return np.transpose(x, (1, 2, 0))[None, ...]
            # Otherwise assume THW (grayscale with time)
            return x[..., None]
        elif nd == 2:
            # single grayscale frame
            return x[None, ..., None]
        else:
            raise ValueError(f"Unsupported ndim={nd} for shape {x.shape}")

    # Normalize to THWC
    thwc = to_thwc(arr, order)

    # Optional truncation on T
    if length != -1:
        thwc = thwc[:length]

    # Validate channels and drop alpha if present
    C = thwc.shape[-1]
    if C not in (1, 3, 4):
        raise ValueError(f"Unsupported channel count C={C}; expected 1, 3, or 4 after normalization.")
    if C == 4:
        thwc = thwc[..., :3]
        C = 3

    # Convert to grayscale (T,H,W)
    if C == 3:
        # Ensure proper float scaling to [0,1] before rgb2gray
        rgb = img_as_float32(thwc)     # keeps shape (T,H,W,3), float64 in [0,1]
        gray = rgb2gray(rgb)         # (T,H,W), float64 in [0,1]
    else:  # C == 1
        gray = img_as_float32(thwc[..., 0])  # (T,H,W), float64 in [0,1]

    # Optional Gaussian blur per frame
    if gaussian_filtered:
        filtered = np.empty_like(gray)
        for t in range(gray.shape[0]):
            filtered[t] = gaussian_filter(gray[t], sigma=2)
        gray = filtered

    if full_channels:
        video = gray[None, :, None, :, :]  # (1, T, 1, H, W)
        video = np.repeat(video, 3, axis=2) #(1, T, 3, H, W)
    else:
        video = gray

    frame_stack = [gray[t] for t in range(gray.shape[0])]

    return video.astype(out_dtype, copy=False), frame_stack, filename

def get_all_paths(input_folder: str) -> list:
    """Get all .tif file paths in the input folder and its subfolders.

    :param input_folder: path to the input folder
    :return: list of file paths
    """
    p = Path(input_folder)
    paths = list(p.rglob("*.tif"))
    return [p for p in paths if p.is_file()]

def ensure_thw(video: np.ndarray) -> np.ndarray:
    """
    Ensure the array is shaped (T, H, W).
    - If (H, W) -> (1, H, W)
    - If (T, H, W, C) -> average across channels -> (T, H, W)
    - If already (T, H, W) -> return as-is

    :param video: input video array
    :return: reshaped video array
    :raises ValueError: if input shape is unsupported
    """
    if video.ndim == 2:
        return video[None, ...]
    if video.ndim == 3:
        return video
    if video.ndim == 4:
        # Convert color to grayscale by channel mean (simple and general)
        return video.mean(axis=-1)
    raise ValueError(f"Unsupported video shape {video.shape}. Expected 2D, 3D (T,H,W), or 4D (T,H,W,C).")


def load_video_legacy(path, length=-1, gaussian_filtered=False):
    """legacy version of load_video for compatibility with existing code."""
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
    if length != -1:
        frames = frames[:length]
    video = torch.from_numpy(np.expand_dims(video, axis=0)).float()
    video = video.permute(0, 2, 1, 3, 4).repeat(1, 1, 3, 1, 1).to(device)[:,:len,:,:,:]
    return video, frames, filename