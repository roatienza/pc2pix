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
from loader import read_view_angle
from plyfile import PlyData

sys.path.append("..")
from general_utils import plot_3d_point_cloud

GT_PATH = "../data/shapenet_release/renders"
PRED_PATH = "data/all"
PLY_PATH = "../data/shape_net_core_uniform_samples_2048"
PLOTS_PATH = "plots3d"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Shapnet category or class (chair, airplane, etc)"
    parser.add_argument("--category", default='chair', help=help_)
    help_ = "Split file"
    parser.add_argument("-s", "--split_file", default='data/all_exp.json', help=help_)
    help_ = "Data png files folder"
    parser.add_argument("--data", default=PRED_PATH, help=help_)
    args = parser.parse_args()

    split_file = args.split_file
    js = get_ply(split_file)
    t = 0
    i = 0
    j = 0
    os.makedirs(PLOTS_PATH, exist_ok=True) 
    for i in range(220):
        images = []
        image_paths = []
        for key in js.keys():
            data = js[key]
            test = data['test']
            tag = test[i]
            target_path = os.path.join(PLOTS_PATH, tag + "_" + str(i) + ".png")
            path = os.path.join(PLY_PATH, key)
            ply_file = os.path.join(path, tag + ".ply")

            ply_data = PlyData.read(ply_file)
            points = ply_data['vertex']
            pc = np.vstack([points['x'], points['y'], points['z']]).T
            path = os.path.join(GT_PATH, key)
            path = os.path.join(path, tag)
            view_file = os.path.join(path, "view.txt")
            elev = read_view_angle(view_file, j) * 40.
            azim = read_view_angle(view_file, j, elev=False) * 180.
            elev = elev[0][0]
            azim = azim[0][0]
            fig = plot_3d_point_cloud(pc[:, 0], 
                                      pc[:, 1], 
                                      pc[:, 2],
                                      show=False,
                                      elev=elev,
                                      azim=azim,
                                      colorize='rainbow',
                                      filename=target_path)
            image_paths.append(target_path)
            #    image = np.array(Image.open(target_path)) / 255.0
            #    images.append(image)

        for key in js.keys():
            # key eg 03001627
            data = js[key]
            test = data['test']
            tag = test[i]
            # tag eg fff29a99be0df71455a52e01ade8eb6a 
            path = os.path.join(GT_PATH, key)
            path = os.path.join(path, tag)
            gt_filename = os.path.join(path, 'render_{}_128.png'.format(j))
            image_paths.append(gt_filename)
            print(gt_filename)

        for key in js.keys():
            # key eg 03001627
            data = js[key]
            test = data['test']
            tag = test[i]
            # tag eg fff29a99be0df71455a52e01ade8eb6a 
            path = os.path.join(args.data, key)
            path = os.path.join(path, tag)
            blender_filename = os.path.join(path, 'blender_render_{}_128.png'.format(j))
            image_paths.append(blender_filename)
            print(blender_filename)

        for key in js.keys():
            # key eg 03001627
            data = js[key]
            test = data['test']
            tag = test[i]
            # tag eg fff29a99be0df71455a52e01ade8eb6a 
            path = os.path.join(args.data, key)
            path = os.path.join(path, tag)
            pc2pix_filename = os.path.join(path, 'pc2pix_render_{}_128.png'.format(j))
            image_paths.append(pc2pix_filename)
            print(pc2pix_filename)

        #print(image_paths)
        for path in image_paths:
            image = np.array(Image.open(path)) / 255.0
            images.append(image)

        plot_images(4, 13, images, str(i) + ".png")
        print(str(i), len(test), tag) 


