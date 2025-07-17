"""utils for bimap image registration."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim

#pth = Path("../../data/low_movement/Experiment-746czi")
pth =  Path("../../data/input/strong_movement/Experiment-591czi")

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


def evaluate(corrected_images: list[np.ndarray], template: np.ndarray) -> list[float]:
    """Evaluate the image registration based on the SSIM of the gradient image."""
    ssim_list = [ssim(template, moving, data_range=template.max() - template.min()) for moving in corrected_images]
    return ssim_list


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


def denoise_stack(imgs: list[np.ndarray]) -> list[np.ndarray]:
    imgs8 = [float32_to_uint8(img) for img in imgs]
    return [cv2.bilateralFilter(img8, d=10, sigmaColor=20, sigmaSpace=50) for img8 in imgs8]

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