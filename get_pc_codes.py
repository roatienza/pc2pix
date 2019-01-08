'''Render point clouds from test dataset using pc2pix

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
import tensorflow as tf

import numpy as np
import argparse

import sys
from ptcloud_stacked_ae import PtCloudStackedAE
from general_utils import plot_3d_point_cloud, plot_image, plot_images
from shapenet import get_split
from in_out import load_ply
from loader import read_view_angle
from general_utils import plot_3d_point_cloud, plot_image, plot_images

import os
import datetime
from PIL import Image
import scipy.misc
sys.path.append("evaluation")
from evaluation.utils import get_ply


def norm_angle(angle):
    angle *= 0.5
    angle += 0.5
    return angle

def norm_pc(pc):
    pc = pc / 0.5
    return pc

def render_by_pc2pix(ptcloud_ae, pc2pix, pc, elev_code, azim_code, filename):
    pc_code = ptcloud_ae.encoder.predict(pc)
    noise = np.random.uniform(-1.0, 1.0, size=[1, 128])
    fake_image = pc2pix.generator.predict([noise, pc_code, elev_code, azim_code])
    fake_image *= 0.5
    fake_image += 0.5
    fake_image = fake_image[0]
    scipy.misc.toimage(fake_image, cmin=0.0, cmax=1.0).save(filename)

    # print(fake_image.shape)
    # fake_image = Image.fromarray(fake_image)
    # fake_image.save(filename)
    # plot_image(fake_image, color=True, filename=filename)

PLY_PATH = "data/shape_net_core_uniform_samples_2048"
PC_CODES_PATH = "pc_codes"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 ptcloud_ae model trained ae weights"
    parser.add_argument("-w", "--ptcloud_ae_weights", help=help_)
    help_ = "Shapnet category or class (chair, airplane, etc)"
    parser.add_argument("-a", "--category", default='all', help=help_)
    help_ = "Split file"
    parser.add_argument("-s", "--split_file", default='data/all_exp_norm.json', help=help_)
    help_ = "PLY files folder"
    parser.add_argument("--ply", default=PLY_PATH, help=help_)
    help_ = "pc codes folder"
    parser.add_argument("--pc_codes", default=PC_CODES_PATH, help=help_)
    help_ = "Point cloud code dim"
    parser.add_argument("-p", "--pc_code_dim", default=32, type=int, help=help_)
    args = parser.parse_args()

    batch_size = 32
    pc_code_dim = args.pc_code_dim
    category = args.category

    ptcloud_ae = PtCloudStackedAE(latent_dim=pc_code_dim,
                                  kernel_size=5)
    ptcloud_ae.stop_sources()

    if args.ptcloud_ae_weights:
        print("Loading point cloud ae weights: ", args.ptcloud_ae_weights)
        ptcloud_ae.use_emd = False
        ptcloud_ae.ae.load_weights(args.ptcloud_ae_weights)
    else:
        print("Trained point cloud ae required to pc2pix")
        exit(0)

    js = get_ply(args.split_file)
    os.makedirs(args.pc_codes, exist_ok=True) 
    filename = args.category + "-" + str(pc_code_dim) + "-pc_codes.npy"
    pc_codes_filename = os.path.join(args.pc_codes, filename) 

    steps = 0
    datasets = ('train', 'test')
    for dataset in datasets:
        for key in js.keys():
            # key eg 03001627
            data = js[key]
            tags = data[dataset]
            steps += len(tags)

    print("Complete data len: ", steps)

    fake_pc_codes = None
    start_time = datetime.datetime.now()
    print("Generating fake pc codes...")
    print("Saving pc codes to file: ", pc_codes_filename)
    i = 0
    for dataset in datasets:
        for key in js.keys():
            # key eg 03001627
            data = js[key]
            tags = data[dataset]
            ply_path_main = os.path.join(args.ply, key)
            for tag in tags:
                # tag eg fff29a99be0df71455a52e01ade8eb6a 
                ply_file = os.path.join(ply_path_main, tag + ".ply")
                pc = norm_pc(load_ply(ply_file))
                shape = pc.shape
                pc = np.reshape(pc, [-1, shape[0], shape[1]])
                fake_pc_code = ptcloud_ae.encoder.predict(pc)
                if fake_pc_codes is None:
                    fake_pc_codes = fake_pc_code
                else:
                    fake_pc_codes = np.append(fake_pc_codes, fake_pc_code, axis=0)
                elapsed_time = datetime.datetime.now() - start_time
                i += 1
                pcent = 100. * float(i)/steps
                log = "%0.2f%% of %d [shape: %s] [tag: %s] [time: %s]" % (pcent, steps, fake_pc_codes.shape, tag, elapsed_time)
                print(log)

    print("Saving pc codes to file: ", pc_codes_filename)
    np.save(pc_codes_filename, fake_pc_codes)
