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
sys.path.append("../lib")
sys.path.append("../external")
sys.path.append("..")
from pc2pix import PC2Pix
from ptcloud_stacked_ae import PtCloudStackedAE
from in_out import load_ply
from loader import read_view_angle

import os
import datetime
from PIL import Image
import scipy.misc
from utils import get_ply

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

PLY_PATH = "../data/shape_net_core_uniform_samples_2048"
VIEW_PATH = "../data/shapenet_release/renders"
TARGET_PATH = "data"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator model trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Load discriminator model trained weights"
    parser.add_argument("-d", "--discriminator", help=help_)
    help_ = "Load h5 ptcloud_ae model trained ae weights"
    parser.add_argument("-w", "--ptcloud_ae_weights", help=help_)
    help_ = "Use color images"
    parser.add_argument("-c", "--color", action='store_true', help=help_)
    help_ = "Shapnet category or class (chair, airplane, etc)"
    parser.add_argument("-a", "--category", default='chair', help=help_)
    help_ = "Split file"
    parser.add_argument("-s", "--split_file", default='data/chair_exp.json', help=help_)
    help_ = "Downsample by this factor"
    parser.add_argument("--downsample", default=1, type=int, help=help_)
    help_ = "Data png files folder"
    parser.add_argument("--data", default=TARGET_PATH, help=help_)
    help_ = "PLY files folder"
    parser.add_argument("--ply", default=PLY_PATH, help=help_)
    args = parser.parse_args()

    batch_size = 32
    pc_code_dim = 32
    gw = None
    dw = None
    gen_pc_codes = False
    color = True
    evaluate = None

    if args.discriminator:
        dw = args.discriminator

    if args.generator:
        gw = args.generator

    ptcloud_ae = PtCloudStackedAE(latent_dim=pc_code_dim,
                                  category=args.category,
                                  kernel_size=5)
    ptcloud_ae.stop_sources()

    if args.ptcloud_ae_weights:
        print("Loading point cloud ae weights: ", args.ptcloud_ae_weights)
        ptcloud_ae.use_emd = False
        ptcloud_ae.ae.load_weights(args.ptcloud_ae_weights)
    else:
        print("Trained point cloud ae required to pc2pix")
        exit(0)

    pc2pix = PC2Pix(ptcloud_ae=ptcloud_ae, gw=gw, dw=dw, batch_size=batch_size, color=color, category=args.category)

    split_file = args.split_file
    js = get_ply(split_file)
    start_time = datetime.datetime.now()
    for key in js.keys():
        # key eg 03001627
        # category_path = os.path.join(OBJ_PATH, key)
        view_path_main = os.path.join(VIEW_PATH, key)
        ply_path_main = os.path.join(args.ply, key)
        target_path_main = os.path.join(args.data, key)
        os.makedirs(target_path_main, exist_ok=True) 
        data = js[key]
        test = data['test']
        for tag in test:
            # tag eg fff29a99be0df71455a52e01ade8eb6a 
            view_path = os.path.join(view_path_main, tag)
            target_path = os.path.join(target_path_main, tag)
            os.makedirs(target_path, exist_ok=True) 
            view_file = os.path.join(view_path, "view.txt")
            ply_file = os.path.join(ply_path_main, tag + ".ply")
            pc = norm_pc(load_ply(ply_file))
            if args.downsample > 1:
                pc = pc[0::args.downsample,:]
                pc = np.repeat(pc, args.downsample, axis=0)
            shape = pc.shape
            pc = np.reshape(pc, [-1, shape[0], shape[1]])
            for i in range(20):
                elev_code = norm_angle(read_view_angle(view_file, i))
                azim_code = norm_angle(read_view_angle(view_file, i, elev=False))
                filename = os.path.join(target_path, 'pc2pix_render_{}_128.png'.format(i))
                render_by_pc2pix(ptcloud_ae, pc2pix, pc, elev_code, azim_code, filename)

            elapsed_time = datetime.datetime.now() - start_time
            print(ply_file, "-->", filename, "Elapsed: ", elapsed_time)

    pc2pix.stop_sources()
    del pc2pix

