# encoding: utf-8
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import sort
import pandas as pd
import numpy as np

DEFAULTS = {
        "model_path": './model_h5/yolo.h5',
        "anchors_path": './model_data/yolo_anchors.txt',
        "classes_path": './model_data/coco_classes.txt',
        "gpu_num": 1,
        "image": False,
        "tracker": True,
        "write_to_file": True,
        "input": './input/Demo3.mp4',
        "output1": './output/Demo3.mp4',
        "output_path": './output/',
        "score": 0.4,  # threshold
        "iou": 0.4,  # threshold
        "repeat_iou": 0.95,  # threshold
    }


def getvalue(FLAGS, defaults):

    args = vars(FLAGS)

    for value in defaults:
        args[value] = defaults[value]

    return FLAGS


def detect_img(yolo):
    while True:

        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            # Initialization
            mot_tracker = sort.Sort()
            yolo.mot_tracker = mot_tracker
            yolo.frame = 1

            if yolo.write_to_file:
                emptyFile = open(yolo.output_path + 'result_no_smooth.dat', 'w')
            else:
                emptyFile = None
            r_image = yolo.detect_image(image, emptyFile)
            if yolo.write_to_file:
                emptyFile.close()
            r_image.save(yolo.__dict__['output_path'] + 'output.png', 'png')
    yolo.close_session()


FLAGS = None

if __name__ == '__main__':

    FLAGS = argparse.Namespace()
    FLAGS = getvalue(FLAGS, DEFAULTS)

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output1)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")