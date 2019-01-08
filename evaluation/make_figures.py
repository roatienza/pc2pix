'''Generate figures

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse

import sys
import os
import datetime
from PIL import Image
import scipy.misc
from utils import get_ply, plot_images

GT_PATH = "../data/shapenet_release/renders"
PRED_PATH = "data"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Shapnet category or class (chair, airplane, etc)"
    parser.add_argument("--category", default='chair', help=help_)
    help_ = "Split file"
    parser.add_argument("-s", "--split_file", default='data/chair_exp.json', help=help_)
    help_ = "Data png files folder"
    parser.add_argument("--data", default=PRED_PATH, help=help_)
    args = parser.parse_args()

    split_file = args.split_file
    js = get_ply(split_file)
    variations = ("2", "4", "0.05", "0.1")
    t = 0
    for key in js.keys():
        # key eg 03001627
        gt_path_main = os.path.join(GT_PATH, key)
        paths = [os.path.join(args.data, key)]
        for v in variations:
            path = os.path.join(args.data, v)
            path = os.path.join(path, key)
            paths.append(path)

        data = js[key]
        test = data['test']
        test_len = len(test)
        for tag in test:
            images = []
            image_paths = []
            # tag eg fff29a99be0df71455a52e01ade8eb6a 
            i = 0
            for p in paths:
                path =  os.path.join(p, tag)
                blender_filename = os.path.join(path, 'blender_render_{}_128.png'.format(i))
                image_paths.append(blender_filename)
            for p in paths:
                path =  os.path.join(p, tag)
                pc2pix_filename = os.path.join(path, 'pc2pix_render_{}_128.png'.format(i))
                image_paths.append(pc2pix_filename)

            for path in image_paths:
                image = np.array(Image.open(path)) / 255.0
                images.append(image)

            t += 1
            plot_images(2, 5, images, tag + ".png")
            print(str(t), len(test), tag) 


