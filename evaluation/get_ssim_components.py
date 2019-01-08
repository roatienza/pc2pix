'''Calculate SSIM and its components 
(Luminance, Contrast, Structure)

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import tensorflow as tf
#tf.enable_eager_execution()

import numpy as np
import argparse

import sys
#sys.path.append("../lib")
#sys.path.append("../external")
#sys.path.append("..")

import os
import datetime
#from PIL import Image
import scipy.misc
from msssim import ssim
from utils import get_ply


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
    b_ssim = []
    b_lumi = []
    b_cont = []
    b_stru = []
    p_ssim = []
    p_lumi = []
    p_cont = []
    p_stru = []
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
            gt = None
            bl = None
            pc = None
            for i in range(20):
                gt_filename = os.path.join(gt_path, 'render_{}_128.png'.format(i))
                blender_filename = os.path.join(pred_path, 'blender_render_{}_128.png'.format(i))
                pc2pix_filename = os.path.join(pred_path, 'pc2pix_render_{}_128.png'.format(i))

                gt_im = scipy.misc.imread(gt_filename)
                w = gt_im.shape[0]
                h = gt_im.shape[1]
                c = gt_im.shape[2]
                gt_im = np.reshape(gt_im, [1, w, h, c])
                bl_im = scipy.misc.imread(blender_filename)
                bl_im = np.reshape(bl_im, [1, w, h, c])
                pc_im = scipy.misc.imread(pc2pix_filename)
                pc_im = np.reshape(pc_im, [1, w, h, c])
                if gt is None:
                    gt = np.array(gt_im)
                    bl = np.array(bl_im)
                    pc = np.array(pc_im)
                else:
                    gt = np.append(gt, gt_im, axis=0)
                    bl = np.append(bl, bl_im, axis=0)
                    pc = np.append(pc, pc_im, axis=0)

            l, c, s, _ssim = ssim(gt, bl)
            b_ssim = np.append(b_ssim, _ssim)
            b_lumi = np.append(b_lumi, l)
            b_cont = np.append(b_cont, c)
            b_stru = np.append(b_stru, s)

            l, c, s, _ssim = ssim(gt, pc)
            p_ssim = np.append(p_ssim, _ssim)
            p_lumi = np.append(p_lumi, l)
            p_cont = np.append(p_cont, c)
            p_stru = np.append(p_stru, s)

            t += 1
            elapsed_time = datetime.datetime.now() - start_time
            print(str(t), "/", test_len, ": ", blender_filename, pc2pix_filename, "Elapsed :", elapsed_time)
            print("b_ssim:", np.mean(b_ssim), "p_ssim:", np.mean(p_ssim))
            print("b_lumi:", np.mean(b_lumi), "p_lumi:", np.mean(p_lumi))
            print("b_cont:", np.mean(b_cont), "p_cont:", np.mean(p_cont))
            print("b_stru:", np.mean(b_stru), "p_stru:", np.mean(p_stru))

