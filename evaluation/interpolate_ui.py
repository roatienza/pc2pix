'''Render point clouds from test dataset using pc2pix

python3 interpolate_ui.py 

python3 interpolate_ui.py --ptcloud_ae_weights=../model_weights/ptcloud/chair-pt-cloud-stacked-ae-chamfer-5-ae-weights-32.h5 -k=5 --generator=../model_weights/pc2pix/chair-gen-color.h5 --discriminator=../model_weights/pc2pix/chair-dis-color.h5  --category="chair" --split_file=data/chair_exp.json -p=32


'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
import tensorflow as tf

import numpy as np
import argparse

import sys
sys.path.append("..")
sys.path.append("../lib")
sys.path.append("../external")
from pc2pix import PC2Pix
from ptcloud_stacked_ae import PtCloudStackedAE
from general_utils import plot_3d_point_cloud
from shapenet import get_split
from in_out import load_ply
from loader import read_view_angle
from general_utils import plot_3d_point_cloud, plot_image, plot_images

import os
import datetime
from PIL import Image
import scipy.misc
#sys.path.append("evaluation")
from utils import get_ply, plot_images

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import time


class Window(Frame):
    # Define settings upon initialization. Here you can specify
    def __init__(self,
                 master,
                 ptcloud_ae,
                 pc2pix,
                 js,
                 ply):
        Frame.__init__(self, master)   
        self.master = master
        self.ptcloud_ae = ptcloud_ae
        self.pc2pix = pc2pix
        self.js = js
        self.ply = ply
        self.pc_codes = []
        self.init_window()
        self.init_pc2pix()
        self.images = []

    #Creation of init_window
    def init_window(self):
        # changing the title of our master widget      
        self.master.title("GUI")
        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)
        self.delta = DoubleVar()
        self.scale = Scale(self.master,
                           variable=self.delta,
                           orient=HORIZONTAL,
                           length=512,
                           command=self.render_images)
        self.scale.pack(anchor=CENTER)


    def render_images(self, value):
        print("Value: ", value)
        if len(self.pc_codes)==0:
            return
        delta = (self.pc_codes[1] - self.pc_codes[0])/(101)
        delta *= (int(value) + 1)
        pc_code = self.pc_codes[0] + delta
        pc = ptcloud_ae.decoder.predict(pc_code)
        pc *= 0.5
        target_path = os.path.join(PLOTS_PATH, value + ".png")
        fig = plot_3d_point_cloud(pc[0][:, 0],
                                  pc[0][:, 1],
                                  pc[0][:, 2],
                                  show=False,
                                  azim=320,
                                  colorize='rainbow',
                                  filename=target_path)
        if len(self.images)==0:
            image = self.show_image(target_path, x=256, y=0)
            self.images.append(image)
            image = render_by_pc2pix(pc_code, self.pc2pix, azim=-40)
            image = self.display_image(image, x=256, y=256)
            self.images.append(image)
        else:
            img = Image.open(target_path)
            render = ImageTk.PhotoImage(img)
            self.images[0].configure(image=render)
            self.images[0].image = render

            img = render_by_pc2pix(pc_code, self.pc2pix, azim=-40)
            img = Image.fromarray(np.uint8(img*255))
            img = img.resize((256, 256), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(img)
            self.images[1].configure(image=render)
            self.images[1].image = render

    def show_image(self, path, x=0, y=0):
        img = Image.open(path)
        render = ImageTk.PhotoImage(img)

        # labels can be text or images
        image = Label(self, image=render)
        image.image = render
        image.place(x=x, y=y)
        return image


    def display_image(self, img, x=0, y=0):
        img = Image.fromarray(np.uint8(img*255))
        img = img.resize((256, 256), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(img)
        # labels can be text or images
        image = Label(self, image=render)
        image.image = render
        image.place(x=x, y=y)
        return image


    def init_pc2pix(self):
        # datasets = ('test')
        os.makedirs(PLOTS_PATH, exist_ok=True)
        # sofa2car - modify these 2 keys manually
        #keys = [ "04256520", "02958343"]
        # sofa2chair 
        keys = [ "04256520", "03001627"]
        # chair2chair
        # keys = [ "03001627", "03001627"]
        tags = []
        data = js[keys[0]]
        tag = data['test']
        tags.append(tag)
        data = js[keys[1]]
        tag = data['test']
        tags.append(tag)
        plys =[]
        ply_path = os.path.join(self.ply, keys[0])
        plys.append(ply_path)
        ply_path = os.path.join(self.ply, keys[1])
        plys.append(ply_path)
        tagslen = min(len(tags[0]), len(tags[1]))

        self.tags = []
        self.pc_codes = []

        np.random.seed(int(time.time()))
        for i in range(2):
            j = np.random.randint(0, tagslen, 1)[0]
            tag = tags[i][j]
            self.tags.append(tag)
            # images = []
            # pc_codes = []
            ply_file = os.path.join(plys[i], tag + ".ply")
            pc = load_ply(ply_file)

            target_path = os.path.join(PLOTS_PATH, tag + ".png")
            fig = plot_3d_point_cloud(pc[:, 0],
                                      pc[:, 1],
                                      pc[:, 2],
                                      show=False,
                                      azim=320,
                                      colorize='rainbow',
                                      filename=target_path)
        
            pc = norm_pc(pc)
            shape = pc.shape
            pc = np.reshape(pc, [-1, shape[0], shape[1]])
            pc_code = ptcloud_ae.encoder.predict(pc)
            self.pc_codes.append(pc_code)

            self.show_image(target_path, x=(i*2)*256, y=0)
            image = render_by_pc2pix(pc_code, self.pc2pix, azim=-40)
            print(image.shape)
            self.display_image(image, x=(i*2)*256, y=256)

        #image = np.array(Image.open(target_path)) / 255.0
        #images.append(image)
        #pc = norm_pc(pc)
        #shape = pc.shape
        #pc = np.reshape(pc, [-1, shape[0], shape[1]])



def render_by_pc2pix(pc_code, pc2pix, elev=10., azim=240.):
    elev += 40.
    azim += 180.
    elev_code = np.array([elev / 80.])
    azim_code = np.array([azim / 360.])
    noise = np.random.uniform(-1.0, 1.0, size=[1, 128])
    fake_image = pc2pix.generator.predict([noise, pc_code, elev_code, azim_code])
    fake_image *= 0.5
    fake_image += 0.5
    fake_image = fake_image[0]
    return fake_image



def norm_angle(angle):
    angle *= 0.5
    angle += 0.5
    return angle


def norm_pc(pc):
    pc = pc / 0.5
    return pc


PLY_PATH = "../data/shape_net_core_uniform_samples_2048"
PC_CODES_PATH = "pc_codes"
PLOTS_PATH = "plots3d"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator model trained weights"
    parser.add_argument("-g",
                        "--generator",
                        default="../model_weights/pc2pix/all-gen-color.h5",
                        help=help_)
    help_ = "Load discriminator model trained weights"
    parser.add_argument("-d",
                        "--discriminator",
                        default="../model_weights/pc2pix/all-dis-color.h5",
                        help=help_)
    help_ = "Load h5 ptcloud_ae model trained ae weights"
    parser.add_argument("-w",
                        "--ptcloud_ae_weights",
                        default="../model_weights/ptcloud/all-pt-cloud-stacked-ae-emd-5-ae-weights-512.h5",
                        help=help_)
    help_ = "Shapnet category or class (chair, airplane, etc)"
    parser.add_argument("-a",
                        "--category",
                        default='chair',
                        help=help_)
    help_ = "Split file"
    parser.add_argument("-s",
                        "--split_file",
                        default='data/all_exp.json',
                        help=help_)
    help_ = "PLY files folder"
    parser.add_argument("--ply", default=PLY_PATH, help=help_)
    help_ = "pc codes folder"
    parser.add_argument("--pc_codes", default=PC_CODES_PATH, help=help_)
    help_ = "Point cloud code dim"
    parser.add_argument("-p", "--pc_code_dim", default=512, type=int, help=help_)
    help_ = "Kernel size"
    parser.add_argument("-k", "--kernel_size", default=5, type=int, help=help_)
    args = parser.parse_args()

    batch_size = 32
    pc_code_dim = args.pc_code_dim
    category = args.category

    ptcloud_ae = PtCloudStackedAE(latent_dim=args.pc_code_dim,
                                  kernel_size=args.kernel_size,
                                  category=category,
                                  evaluate=True)
    # ptcloud_ae.stop_sources()

    if args.ptcloud_ae_weights:
        print("Loading point cloud ae weights: ", args.ptcloud_ae_weights)
        ptcloud_ae.use_emd = False
        ptcloud_ae.ae.load_weights(args.ptcloud_ae_weights)
    else:
        print("Trained point cloud ae required to pc2pix")
        exit(0)

    pc2pix = PC2Pix(ptcloud_ae=ptcloud_ae,
                    gw=args.generator,
                    dw=args.discriminator, 
                    pc_code_dim=args.pc_code_dim, 
                    batch_size=batch_size, 
                    category=category)

    js = get_ply(args.split_file)

    root = Tk()
    root.geometry("768x532")
    app = Window(root, ptcloud_ae=ptcloud_ae, pc2pix=pc2pix, js=js, ply=args.ply)
    root.mainloop()
    ptcloud_ae.stop_sources()
