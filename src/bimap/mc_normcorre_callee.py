import sys
import time
from pathlib import Path
import numpy as np
import caiman as cm
from caiman.motion_correction import MotionCorrect

max_shifts = (6, 6)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
overlaps = (24, 24)  # overlap between patches (size of patch strides+overlaps)
max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)


def main():
    # Just echo back the args for demo purposes:
    print("Running in ’caiman’ env!")
    print("Received args:", sys.argv[1:])
    filename = Path(sys.argv[1]).stem
    output_path = sys.argv[2]
    movie = cm.load(sys.argv[1])
    start_time = time.time()
    mc = MotionCorrect(movie, max_shifts=max_shifts,
                       strides=strides, overlaps=overlaps,
                       max_deviation_rigid=max_deviation_rigid,
                       shifts_opencv=shifts_opencv, nonneg_movie=True,
                       border_nan=border_nan)
    mc.motion_correct(save_movie=True)

    mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
    mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)
    mc.split_els = int(700 / 200)
    mc.split_rig = int(700 / 200)

    mc.motion_correct(save_movie=True, template=mc.total_template_rig)
    m_els = cm.load(mc.fname_tot_els)
    end_time = time.time()
    m_els.save(
        f'{output_path}/{filename}.tif')
    print(f"TIME_TAKEN={end_time - start_time}")
    print("Done!")


if __name__ == "__main__":
    main()