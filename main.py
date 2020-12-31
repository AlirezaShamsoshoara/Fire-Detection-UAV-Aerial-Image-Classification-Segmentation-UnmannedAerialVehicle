"""
Created on Aug. 12, 2020
@author: Alireza Shamsoshoara
@Project: Aerial image dataset for fire classification, segmentation, and scheduling using
          Unmanned Aerial Vehicles (UAVs)
          Paper: ### TODO: WILL UPDATE HERE AFTER ACCEPTANCE ...
          Arxiv: https://arxiv.org/pdf/2012.14036.pdf
          Dataset: https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs
          YouTube Video: https://www.youtube.com/watch?v=bHK6g37_KyA
@Northern Arizona University
This project is developed and tested with Python 3.6 using pycharm on Ubuntu 18.04 LTS machine
"""

#################################
# Main File
#################################

# ############# import libraries
# General Modules

# Customized Modules
from config import Mode
from config import Flags

from config import pathVid_fire
from config import pathFrame_all
from config import pathFrame_resize
from config import pathVid_LakeMary
from config import pathFrame_lakemary
from config import pathFrame_resize_lakemary

from config import pathFrame_test
from config import pathVid_test_Fire
from config import pathVid_test_NoFire
from config import pathFrame_resize_test

from utils import resize
from utils import get_fps
from utils import play_vid
from utils import vid_to_frame
from utils import rename_all_files

from training import train_keras
from classification import classify
from segmentation import segmentation_keras_load

# from plotdata import plot_scheduling
# from scheduling import uav_scheduling


def main():
    fps = get_fps(path_vid)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    if Flags.get('playVideoFlag'):
        play_vid(path_vid)
    if Flags.get('SaveRawFrameFlag'):
        vid_to_frame(path_vid, mode=Mode)
    if Flags.get('ResizeFlag'):
        resize(path_load, path_save_resize, mode=Mode)


if __name__ == "__main__":
    if Mode == 'Fire':
        path_vid = pathVid_fire
        path_load = pathFrame_all
        path_save_resize = pathFrame_resize
        main()
    elif Mode == 'Lake_Mary':
        path_vid = pathVid_LakeMary
        path_load = pathFrame_lakemary
        path_save_resize = pathFrame_resize_lakemary
        main()
    elif Mode == 'Test_Frame':
        # path_vid = pathVid_test_Fire
        path_vid = pathVid_test_NoFire
        path_load = pathFrame_test
        path_save_resize = pathFrame_resize_test
        main()
    elif Mode == 'Training':
        train_keras()
    elif Mode == 'Classification':
        classify()
    elif Mode == 'Rename':
        rename_all_files(path="Image")
        rename_all_files(path="Mask")
    elif Mode == 'Segmentation':
        segmentation_keras_load()
    # elif Mode == 'Scheduling':
    #     uav_scheduling()
    #     plot_scheduling()
    else:
        print("Mode is not correct")
