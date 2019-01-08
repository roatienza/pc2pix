'''Using Inception V3, determine the rendered object class

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
#from inception import get_class_confidence
from utils import get_ply
import inception

GT_PATH = "../data/shapenet_release/renders"
PRED_PATH = "data"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Shapenet category or class (chair, airplane, etc)"
    parser.add_argument("--category", default='chair', help=help_)
    help_ = "Split file"
    parser.add_argument("-s", "--split_file", default='data/chair_exp.json', help=help_)
    help_ = "Data png files folder"
    parser.add_argument("--data", default=PRED_PATH, help=help_)
    args = parser.parse_args()

    # Download Inception model if not already done.
    inception.maybe_download()

    # Load the Inception model so it is ready for classifying images.
    model = inception.Inception()

    split_file = args.split_file
    js = get_ply(split_file)
    obj_class = {}
    obj_class['render'] = {}
    obj_class['pc2pix'] = {}
    obj_class['blender'] = {}
    start_time = datetime.datetime.now()
    t = 0
    for key in js.keys():
        # key eg 03001627
        gt_path_main = os.path.join(GT_PATH, key)
        pred_path_main = os.path.join(args.data, key)
        data = js[key]
        test = data['test']
        test_len = len(test)
        for tag in test:
            # tag eg fff29a99be0df71455a52e01ade8eb6a 
            gt_path = os.path.join(gt_path_main, tag)
            pred_path = os.path.join(pred_path_main, tag)
            t += 1 
            for i in range(20):
                gt_filename = os.path.join(gt_path, 'render_{}_128.png'.format(i))
                blender_filename = os.path.join(pred_path, 'blender_render_{}_128.png'.format(i))
                pc2pix_filename = os.path.join(pred_path, 'pc2pix_render_{}_128.png'.format(i))
                paths = (gt_filename, blender_filename, pc2pix_filename)
                for path in paths:
                    if path == gt_filename:
                        k = 'render'
                    elif path == blender_filename:
                        k = 'blender'
                    else:
                        k = 'pc2pix'
                    pred = model.classify(image_path=path)
                    name, score = model.get_class_scores(pred=pred)
                    if 'chair' in name or 'throne' in name:
                        if name in obj_class[k].keys():
                            obj_class[k][name] += 1
                        else:
                            obj_class[k][name] = 1

            for k in obj_class.keys():
                print(str(t), "/", test_len, k, obj_class[k])
    model.close()
