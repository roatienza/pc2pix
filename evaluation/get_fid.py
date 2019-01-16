'''Calculate the FID between ground truth
render and predicted renders

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import argparse

import sys
# git clone https://github.com/bioinf-jku/TTUR.git
sys.path.append("TTUR")
import fid

import os
import datetime
import scipy.misc
from msssim import ssim
from utils import get_ply


GT_PATH = "../data/shapenet_release/renders"
PRED_PATH = "data"

inception_path = fid.check_or_download_inception(None) # download inception network

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Use color images"
    parser.add_argument("-c", "--color", action='store_true', help=help_)
    help_ = "Shapnet category or class (chair, airplane, etc)"
    parser.add_argument("--category", default='chair', help=help_)
    help_ = "Split file"
    parser.add_argument("-s", "--split_file", default='data/chair_exp.json', help=help_)
    help_ = "Data png files folder"
    parser.add_argument("--data", default=PRED_PATH, help=help_)
    args = parser.parse_args()

    split_file = args.split_file
    js = get_ply(split_file)
    start_time = datetime.datetime.now()
    t = 0
    gt = []
    bl = []
    pc = []
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
            for i in range(20):
                gt_filename = os.path.join(gt_path, 'render_{}_128.png'.format(i))
                blender_filename = os.path.join(pred_path, 'blender_render_{}_128.png'.format(i))
                pc2pix_filename = os.path.join(pred_path, 'pc2pix_render_{}_128.png'.format(i))

                gt_im = scipy.misc.imread(gt_filename)
                bl_im = scipy.misc.imread(blender_filename)
                pc_im = scipy.misc.imread(pc2pix_filename)
                gt.append(gt_im)
                bl.append(bl_im)
                pc.append(pc_im)

            t += 1
            elapsed_time = datetime.datetime.now() - start_time
            print(str(t), "/", test_len, ": ", blender_filename, pc2pix_filename, "Elapsed :", elapsed_time)
            print(np.array(gt).shape)

    gt = np.array(gt)
    bl = np.array(bl)
    pc = np.array(pc)

    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu_gt, sigma_gt = fid.calculate_activation_statistics(gt, sess)
        mu_bl, sigma_bl = fid.calculate_activation_statistics(bl, sess)
        mu_pc, sigma_pc = fid.calculate_activation_statistics(pc, sess)

    fid_value = fid.calculate_frechet_distance(mu_bl, sigma_bl, mu_gt, sigma_gt)
    filename = "fid.log"
    fd = open(filename, "a+")
    fd.write("---| ")
    fd.write(args.split_file)
    fd.write(" |---\n")
    print("Surface FID: %s" % fid_value)
    fd.write("Surface FID: %s\n" % fid_value)
    fid_value = fid.calculate_frechet_distance(mu_pc, sigma_pc, mu_gt, sigma_gt)
    print("PC2PIX FID: %s" % fid_value)
    fd.write("PC2PIX FID: %s\n" % fid_value)
    fd.write("---\n")
    fd.close()
