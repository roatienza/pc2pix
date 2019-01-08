'''Generate 3d pt cloud plots

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
from render_by_pc2pix import norm_angle
from loader import read_view_angle
from plyfile import PlyData

sys.path.append("..")
from general_utils import plot_3d_point_cloud

PLY_PATH = "../data/shape_net_core_uniform_samples_2048"
VIEW_PATH = "../data/shapenet_release/renders"
PLOTS_PATH = "plots3d"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Shapnet category or class (chair, airplane, etc)"
    parser.add_argument("--category", default='chair', help=help_)
    help_ = "Split file"
    parser.add_argument("-s", "--split_file", default='data/chair_exp.json', help=help_)
    help_ = "PLY files folder"
    parser.add_argument("--ply", default=PLY_PATH, help=help_)
    args = parser.parse_args()

    split_file = args.split_file
    js = get_ply(split_file)
    t = 0
    variations = ("2", "4", "0.05", "0.1")
    os.makedirs(PLOTS_PATH, exist_ok=True) 
    for key in js.keys():
        # key eg 03001627
        view_path_main = os.path.join(VIEW_PATH, key)
        paths = [os.path.join(PLY_PATH, key)]
        for v in variations:
            path = os.path.join("ply", v)
            path = os.path.join(path, key)
            paths.append(path)

        data = js[key]
        test = data['test']
        for tag in test:
            images = []
            # tag eg fff29a99be0df71455a52e01ade8eb6a 
            view_path = os.path.join(view_path_main, tag)
            view_file = os.path.join(view_path, "view.txt")
            i = 0
            elev = read_view_angle(view_file, i) * 40.
            azim = read_view_angle(view_file, i, elev=False) * 180.
            elev = elev[0][0]
            #if elev < 0:
            #    elev += 360.
            azim = azim[0][0]
            #if azim < 0:
            #    azim += 360.
            i = 0
            t += 1
            for path in paths:
                target_path = os.path.join(PLOTS_PATH, tag + "-" + str(i) + ".png")
                ply_file = os.path.join(path, tag + ".ply")
                i += 1

                ply_data = PlyData.read(ply_file)
                points = ply_data['vertex']
                pc = np.vstack([points['x'], points['y'], points['z']]).T
                fig = plot_3d_point_cloud(pc[:, 0], 
                                          pc[:, 1], 
                                          pc[:, 2],
                                          show=False,
                                          elev=elev,
                                          azim=azim,
                                          colorize='rainbow',
                                          filename=target_path)
                #fig.close('all')
                image = np.array(Image.open(target_path)) / 255.0
                images.append(image)
            print(str(t), view_file, ply_file, "-->", target_path, elev, azim)
            plot_images(1, 5, images, tag + ".png", dir_name="point_clouds")



