'''Render point clouds from test dataset using pc2pix


python3 interpolate_diff_objs.py --ptcloud_ae_weights=../model_weights/ptcloud/all-pt-cloud-stacked-ae-emd-5-ae-weights-512.h5 -p=512 -k=5 --generator=../model_weights/pc2pix/all-gen-color.h5 --discriminator=../model_weights/pc2pix/all-dis-color.h5  --category="all" --split_file=data/all_exp.json

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
        # sofa2car
        keys = [ "04256520", "02958343"]
        # keys = [ "03001627", "04379243", "04256520"  ]
        # key eg 03001627
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
    ptcloud_ae.stop_sources()

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

    exit(0)

    datasets = ('test')
    start_time = datetime.datetime.now()
    os.makedirs(PLOTS_PATH, exist_ok=True)
    t = 0
    interpolate = True
    # sofa2car
    # keys = [ "04256520", "02958343"]
    keys = [ "03001627", "04379243", "04256520"  ]
    # key eg 03001627
    data = js[keys[0]]
    tags1 = data['test']
    data = js[keys[1]]
    tags2 = data['test']
    data = js[keys[2]]
    tags3 = data['test']
    ply_path_main1 = os.path.join(args.ply, keys[0])
    ply_path_main2 = os.path.join(args.ply, keys[1])
    ply_path_main3 = os.path.join(args.ply, keys[2])
    tagslen = min(len(tags1), len(tags2))
    tagslen = min(tagslen, len(tags3))
    n_interpolate = 10
    if not interpolate:
        n_interpolate = 2
    for _ in range(tagslen):
        n = 0
        i = np.random.randint(0, tagslen, 1)[0]
        tag = tags1[i]
        images = []
        pc_codes = []
        ply_file = os.path.join(ply_path_main1, tag + ".ply")
        pc = load_ply(ply_file)
        target_path = os.path.join(PLOTS_PATH, tag + "_" + str(n) + ".png")
        n += 1
        fig = plot_3d_point_cloud(pc[:, 0],
                                  pc[:, 1],
                                  pc[:, 2],
                                  show=False,
                                  azim=320,
                                  colorize='rainbow',
                                  filename=target_path)
        image = np.array(Image.open(target_path)) / 255.0
        images.append(image)
        pc = norm_pc(pc)
        shape = pc.shape
        pc = np.reshape(pc, [-1, shape[0], shape[1]])
        pc_code1 = ptcloud_ae.encoder.predict(pc)
        pc_codes.append(pc_code1)

        i = np.random.randint(0, tagslen, 1)[0]
        tag = tags2[i]
        ply_file = os.path.join(ply_path_main2, tag + ".ply")
        pc = load_ply(ply_file)
        target_path = os.path.join(PLOTS_PATH, tag + "_" + str(n_interpolate + 1) + ".png")
        fig = plot_3d_point_cloud(pc[:, 0],
                                  pc[:, 1],
                                  pc[:, 2],
                                  azim=320,
                                  show=False,
                                  colorize='rainbow',
                                  filename=target_path)

        image_end = np.array(Image.open(target_path)) / 255.0
        pc = norm_pc(pc)
        shape = pc.shape
        pc = np.reshape(pc, [-1, shape[0], shape[1]])
        pc_code2 = ptcloud_ae.encoder.predict(pc)

        shape = pc_code1.shape
        if interpolate:
            for i in range(n_interpolate):
                #pc_code = []
                delta = (pc_code2 - pc_code1)/(n_interpolate + 1)
                delta *= (i + 1)
                pc_code = pc_code1 + delta
                pc_codes.append(pc_code)

                pc = ptcloud_ae.decoder.predict(pc_code)
                pc *= 0.5
                target_path = os.path.join(PLOTS_PATH, tag + "_" + str(n) + ".png")
                n += 1
                fig = plot_3d_point_cloud(pc[0][:, 0],
                                          pc[0][:, 1],
                                          pc[0][:, 2],
                                          show=False,
                                          azim=320,
                                          colorize='rainbow',
                                          filename=target_path)
                image = np.array(Image.open(target_path)) / 255.0
                images.append(image)

            images.append(image_end)
            pc_codes.append(pc_code2)
        else:
            i = np.random.randint(0, tagslen, 1)[0]
            tag = tags3[i]
            ply_file = os.path.join(ply_path_main3, tag + ".ply")
            pc = load_ply(ply_file)
            target_path = os.path.join(PLOTS_PATH, tag + "_" + str(1) + ".png")
            fig = plot_3d_point_cloud(pc[:, 0],
                                      pc[:, 1],
                                      pc[:, 2],
                                      show=False,
                                      azim=320,
                                      colorize='rainbow',
                                      filename=target_path)

            image = np.array(Image.open(target_path)) / 255.0
            images.append(image)
            pc = norm_pc(pc)
            shape = pc.shape
            pc = np.reshape(pc, [-1, shape[0], shape[1]])
            pc_code = ptcloud_ae.encoder.predict(pc)
            pc_codes.append(pc_code)


            images.append(image_end)
            pc_codes.append(pc_code2)

            pc_code = pc_code1 - pc_code + pc_code2
            pc_codes.append(pc_code)
            pc = ptcloud_ae.decoder.predict(pc_code)
            pc *= 0.5
            target_path = os.path.join(PLOTS_PATH, tag + "_" + str(3) + ".png")
            n += 1
            fig = plot_3d_point_cloud(pc[0][:, 0],
                                      pc[0][:, 1],
                                      pc[0][:, 2],
                                      show=False,
                                      azim=320,
                                      colorize='rainbow',
                                      filename=target_path)
            image = np.array(Image.open(target_path)) / 255.0
            images.append(image)

        for pc_code in pc_codes:
            # default of plot_3d_point_cloud is azim=240 which is -120
            # or 60 = 180 - 120
            image = render_by_pc2pix(pc_code, pc2pix, azim=-40)
            images.append(image)

        print(len(images))
        plot_images(2, n_interpolate + 2, images, tag + ".png", dir_name="point_clouds")
        t += 1
        if t > 2:
            del pc2pix
            del ptcloud_ae
            exit(0)
        #exit(0)
