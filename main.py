import cv2
import numpy as np
from aicspylibczi import CziFile
from pathlib import Path
import matplotlib.pyplot as plt
import ants

pth = Path('./data/low_movement/Experiment-746czi')


def main():
    fixed_path = pth / 'frame_0000.tif'
    moved_path = pth / 'frame_0400.tif'
    fixed = ants.image_read(fixed_path.absolute().as_posix())
    moved = ants.image_read(moved_path.absolute().as_posix())
    fixed.plot(overlay=moved, title='Before Registration')
    mytx = ants.registration(fixed=fixed, moving=moved, type_of_transform='SyN')
    print(mytx)
    warped_moving = mytx['warpedmovout']
    fixed.plot(overlay=warped_moving, title='After Registration')


if __name__ == "__main__":
    main()
