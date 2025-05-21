from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from scipy.signal import correlate
import matplotlib.pyplot as plt
from tqdm import tqdm


pth = Path('./data/low_movement/Experiment-746czi')


def find_highest_correlation(frame_stack: np.array, plot=False):
    mean_corrs = []
    for i in tqdm(range(frame_stack.shape[0]-1)):
        corr_2d = correlate(frame_stack[i], frame_stack[i+1], method='auto')
        mean_corrs.append(np.mean(corr_2d))
    if plot:
        plt.plot(mean_corrs)
        plt.show()
    return int(np.argmax(mean_corrs))


def main():
    frame_paths = list(pth.glob('frame_*.tif'))
    frame_paths = [x.as_posix() for x in frame_paths]
    frame_paths.sort()
    frames = [np.array(Image.open(path)) for path in frame_paths]
    frames = np.asarray(frames)
    template_index = find_highest_correlation(frames, plot=False)
    print(f'{template_index=}')

if __name__ == "__main__":
    main()
