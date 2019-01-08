import numpy as np
import argparse
from PIL import Image
import sys
import os
import datetime

from utils import get_ply

sys.path.append("..")
import scipy.io as sio

OBJ_PATH = "data"
DATA_PATH = "../data/shapenet_release/renders"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Split file"
    parser.add_argument("-s", "--split_file", default='data/chair_exp.json', help=help_)
    help_ = "Data obj files folder"
    parser.add_argument("--data", default=OBJ_PATH, help=help_)
    args = parser.parse_args()

    split_file = args.split_file
    js = get_ply(split_file)
    start_time = datetime.datetime.now()
    for key in js.keys():
        # key eg 03001627
        category_path = os.path.join(args.data, key)
        cam_path = os.path.join(DATA_PATH, key)
        data = js[key]
        test = data['test']
        for tag in test:
            # tag eg fff29a99be0df71455a52e01ade8eb6a 
            object_path = os.path.join(category_path, tag)
            render_path = object_path
            object_path = os.path.join(object_path, tag + ".obj")
            camera_path = os.path.join(cam_path, tag)
            for i in range(20):
                camera = "camera_" + str(i) + ".mat"
                camera_mat = os.path.join(camera_path, camera)
                pos = sio.loadmat(camera_mat)
                pos = list(pos['pos'][0])
                pos = [pos[1], -pos[0], pos[2]]
                # print(pos)
                # print(camera_mat)
                # print(object_path)
                cmd = "blender --background --python render_blender.py -- " + object_path
                cmd += " --x=" + str(pos[0]) + " --y=" + str(pos[1]) + " --z=" + str(pos[2]) 
                cmd += " --i=" + str(i)
                cmd += " --output_folder=" + render_path
                render_filepath = os.path.join(render_path, 'blender_render_{}_128.png'.format(i))
                os.system(cmd)
                img = Image.open(render_filepath)
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                bg.save(render_filepath)
                img.close()
                elapsed_time = datetime.datetime.now() - start_time
                print(render_filepath, "Elapsed: ", elapsed_time)
