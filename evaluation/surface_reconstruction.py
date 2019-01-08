import numpy as np
import argparse

import sys
import os
import datetime
from utils import get_ply

sys.path.append("..")
from shapenet import get_split

PLY_PATH = "../data/shape_net_core_uniform_samples_2048"
DATA_PATH = "data"
MLX = "viewer/ball-pivoting-close-holes-mc-rotate.mlx"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Split file"
    parser.add_argument("-s", "--split_file", default='data/chair_exp.json', help=help_)
    help_ = "PLY files folder"
    parser.add_argument("--ply", default=PLY_PATH, help=help_)
    help_ = "Data obj files folder"
    parser.add_argument("--data", default=DATA_PATH, help=help_)
    args = parser.parse_args()

    os.makedirs(args.data, exist_ok=True) 
    split_file = args.split_file
    js = get_ply(split_file)
    start_time = datetime.datetime.now()
    i = 0
    for key in js.keys():
        target_path = os.path.join(args.data, key)
        os.makedirs(target_path, exist_ok=True) 
        data = js[key]
        test = data['test']
        source_path = os.path.join(args.ply, key)
        for tag in test:
            source_ply = os.path.join(source_path, tag + ".ply")
            target_obj = os.path.join(target_path, tag)
            os.makedirs(target_obj, exist_ok=True) 
            target_obj = os.path.join(target_obj, tag + ".obj")
            cmd = "meshlabserver -i " + source_ply + " -o " + target_obj + " -s " + MLX + " -om vn"
            # print(cmd)
            os.system(cmd)
            elapsed_time = datetime.datetime.now() - start_time
            i += 1
            print(str(i), source_ply, " --> ", target_obj, " Elapsed: ", elapsed_time)



