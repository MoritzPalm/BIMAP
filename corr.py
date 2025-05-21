import numpy as np
import cv2
from scikit-image.registration import phase_cross_correlation
import matplotlib.pyplot as plt


pth = Path('./data/low_movement/Experiment-746czi')


def main():
    all_frames = list(pth.glob('frame_*.tif'))
    all_frames = [x.as_posix() for x in all_frames]
    all_frames.sort()
    for i, frame in enumerate(all_frames):
        corr = phase_cross_correlation(frame, all_frames[i+1], disambiguate=True, )
    


if __name__ == "__main__":
    main()
