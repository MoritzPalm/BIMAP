import time

import numpy as np
import ants

from utils import load_video, save_and_display_video, find_highest_correlation, evaluate

def main():
    path = "../../data/input/strong_movement/b5czi.tif"
    video, frames, filename = load_video(path)
    #template_idx = find_highest_correlation(frames)
    template_idx = 0
    result = run_ants(frames, template_idx, filename=filename)


def run_ants(frame_stack: list[np.ndarray], template_idx: int, filename: str, ants_method: str = "SyNOnly"):
    """Image Registration using the AnTsPy package."""
    motion_corrected_images = []
    fixed = ants.from_numpy(frame_stack[template_idx])
    start_time = time.time()
    for i in range(len(frame_stack)):
        moving = ants.from_numpy(frame_stack[i])
        areg = ants.registration(fixed, moving, ants_method)
        motion_corrected_images.append(areg["warpedmovout"].numpy().astype(np.float32))
    end_time = time.time()
    save_and_display_video(np.array(motion_corrected_images), f'../../data/output/ants/{filename}.mp4')
    metrics = evaluate(motion_corrected_images, frame_stack, frame_stack[template_idx])
    return motion_corrected_images, metrics, end_time - start_time


if __name__ == "__main__":
    main()