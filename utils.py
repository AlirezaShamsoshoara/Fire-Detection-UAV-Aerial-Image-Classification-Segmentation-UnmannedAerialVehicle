"""
#################################
Util functions such as
    1) Playing video
    2) Getting FPS
    3) Extracting Frames
    4) Resizing frames
    5) Renaming files of a directory
#################################
"""

#########################################################
# import libraries

import os
import re
import cv2
from config import new_size


#########################################################
# Function definition

def play_vid(path_vid):
    """
    This function plays the imported vide based on the path.
    :param path_vid: The path of the vide
    :return: None
    """
    cap = cv2.VideoCapture(path_vid)
    while cap.isOpened():
        ret, frame = cap.read()
        color = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        cv2.imshow('frame', color)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def get_fps(path_vid):
    """
    This function return the recorded FPS of the vide.
    :param path_vid: The path of the vide
    :return: The video file's FPS
    """
    cap = cv2.VideoCapture(path_vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    cv2.destroyAllWindows()
    return fps


def vid_to_frame(path_vid, mode):
    """
    Extracting the frames from the video file.
    :param path_vid: The path of the video
    :param mode: Based on the opened file for this project 1) Fire, 2) No-Fire:Lake Mary, 3) Extracting for test set
    :return: None
    """
    vidcap = cv2.VideoCapture(path_vid)
    success, image = vidcap.read()
    count = 0
    while success:
        if mode == 'Fire':
            cv2.imwrite("frames/all/frame%d.jpg" % count, image)  # Save JPG file
        elif mode == 'Lake_Mary':
            cv2.imwrite("frames/lakemary/lake_frame%d.jpg" % count, image)  # Save JPG file
        elif mode == 'Test_Frame':
            # cv2.imwrite("frames/Test_frame/Fire/test_fire_frame%d.jpg" % count, image)  # Save JPG file for FIRE
            cv2.imwrite("frames/Test_frame/No_Fire/test_nofire_frame%d.jpg" % count, image)
            # Save JPG file for NO_FIRE, for NO Fire frames uncomment this line and comment the previous line
        success, image = vidcap.read()
        print('Extract new frame: ', success, ' frame = ', count)
        count += 1


def resize(path_all, path_resize, mode):
    """
    Resizing the imported images to the project and save them on drive based on the dimension parameter.
    :param path_all: The directory of loaded images to the project
    :param path_resize: The directory to save the resized files
    :param mode: Fire, No_Fire(lake mary), or the test data
    :return: None
    """
    image_names_dir = os.listdir(path_all)
    if mode == 'Test_Frame':
        # image_names_dir = os.listdir(path_all + 'Fire')  # This is for the FIRE DIR (Test)
        image_names_dir = os.listdir(path_all + 'No_Fire')  # This is for the No_FIRE DIR (Test)
    image_names_dir.sort()
    new_width = new_size.get('width')
    new_height = new_size.get('height')
    dimension = (new_width, new_height)

    count = 0
    for image in image_names_dir:
        # print(resized_img.shape)
        # cv2.imshow('output', resized_img)
        if mode == 'Fire':
            img = cv2.imread(path_all + '/' + image)
            resized_img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path_resize + '/resized_' + image, resized_img)
        elif mode == 'Lake_Mary':
            img = cv2.imread(path_all + '/' + image)
            resized_img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path_resize + '/lake_resized_' + image, resized_img)
        elif mode == 'Test_Frame':
            # img = cv2.imread(path_all + 'Fire/' + image)
            img = cv2.imread(path_all + 'No_Fire/' + image)
            resized_img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
            # cv2.imwrite(path_resize + 'Fire/resized_' + image, resized_img)  # Resize for Fire (Test Data)
            cv2.imwrite(path_resize + 'No_Fire/resized_' + image, resized_img)  # Resize for NoFire (Test Data)

        print('Image Resized ' + str(count) + ' : resized_' + image)
        count += 1


def rename_all_files(path=None):
    """
    This function returns all the files included in the path directory. This function is used for the fire segmentation
    challenge to have the same name for both the frame and the peered mask.
    :param path: The input directory to rename the included files
    :return: None
    """
    regex = re.compile(r'\d+')
    if path is "Image":
        path_dir = "frames/Segmentation/Data/Images"
    elif path is "Mask":
        path_dir = "frames/Segmentation/Data/Masks"
    else:
        print("Wrong Path for renaming!")
        print("Exit with return")
        return
    files_images = os.listdir(path_dir)
    files_images.sort()
    for count, filename in enumerate(files_images):
        num_ex = regex.findall(filename)
        if path is "Image":
            dst = "image_" + num_ex[0] + ".jpg"
        elif path is "Mask":
            dst = "image_" + str(int(num_ex[0]) - 1) + ".png"
        else:
            print("\nWrong path option ... ")
            return 0
        dst = path_dir + '/' + dst
        src = path_dir + '/' + filename
        os.rename(src, dst)
        print("count = ", count)
