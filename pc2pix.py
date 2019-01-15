'''pc2pix: A conditional generative model for rendering point clouds

Training:
    python3 pc2pix.py --ptcloud_ae_weights=model_weights/ptcloud/all-pt-cloud-stacked-ae-emd-5-ae-weights-512.h5 -t -p=512 --generator=model_weights/pc2pix/all-gen-color.h5 --discriminator=model_weights/pc2pix/all-dis-color.h5

'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
from keras.utils.multi_gpu_utils import multi_gpu_model
import tensorflow as tf


import numpy as np
import argparse

from data import DataSource
from ptcloud_stacked_ae import PtCloudStackedAE
from general_utils import plot_image, plot_images

import sys
import os
import datetime
sys.path.append("lib")
import model


class PC2Pix():
    def __init__(self,
                 ptcloud_ae=None,
                 gw=None,
                 dw=None,
                 pc_code_dim=32,
                 batch_size=64,
                 color=True,
                 gpus=1,
                 category='all'):

        self.noise_dim = 128
        self.ptcloud_ae = ptcloud_ae
        self.gw = gw
        self.dw = dw
        self.gpus = gpus
        self.pc_code_dim = pc_code_dim
        self.category = category
        self.model_dir = "saved_models"
        self.kernel_size = 3
        self.batch_size = batch_size
        self.generator = None
        self.discriminator = None
        self.adversarial = None
        os.makedirs(self.model_dir, exist_ok=True) 
        os.makedirs("weights", exist_ok=True) 
        self.color = color
        self.gen_spectral_normalization = False

        if color:
            # color images 128x128 rgb
            items = ['im_128', 'pc', 'elev', 'azim']
        else:
            # graycale images 224x224
            items = ['gray_128', 'pc', 'elev', 'azim']

            # big color
            # items = ['im', 'pc', 'elev', 'azim']
        # items = ['gray', 'pc', 'view']
        if category == 'all':
            path = 'all_exp_norm.json'
        else:
            path = category + '_exp.json'
        self.split_file = os.path.join('data', path)

        self.train_source = DataSource(batch_size=self.batch_size, items=items, split_file=self.split_file)
        shapenet = self.train_source.dset
        self.epoch_datalen = len(shapenet.get_smids('train')) * shapenet.num_renders
        self.train_steps = self.epoch_datalen // self.batch_size

        pc_codes = "pc_codes"
        path = self.category + "-" + str(pc_code_dim) + "-pc_codes.npy"
        self.pc_codes_filename = os.path.join(pc_codes,path) # "weights/pc_codes.npy"
        self.test_source = DataSource(batch_size=36, smids='test', items=items, nepochs=20, split_file=self.split_file)

        self.build_gan()
        
       
    def generate_fake_pc_codes(self):
        fake_pc_codes = None
        start_time = datetime.datetime.now()
        print("Generating fake pc codes...")
        steps = 4 * self.train_steps
        for i in range(steps):
            _, fake_pc, _, _ = self.train_source.next_batch()
            fake_pc = fake_pc / 0.5
            fake_pc_code = self.ptcloud_ae.encoder.predict(fake_pc)
            if fake_pc_codes is None:
                fake_pc_codes = fake_pc_code
            else:
                fake_pc_codes = np.append(fake_pc_codes, fake_pc_code, axis=0)
            elapsed_time = datetime.datetime.now() - start_time
            pcent = 100. * float(i)/steps
            log = "%0.2f%% [shape: %s] [time: %s]" % (pcent, fake_pc_codes.shape, elapsed_time)
            print(log)

        print("Saving pc codes to file: ", self.pc_codes_filename)
        np.save(self.pc_codes_filename, fake_pc_codes)


    def train_gan(self):
        plot_interval = 500
        save_interval = 500
        start_time = datetime.datetime.now()
        test_image, pc, test_elev_code, test_azim_code = self.test_source.next_batch()
        pc = pc / 0.5
        test_pc_code = self.ptcloud_ae.encoder.predict(pc)
        noise_ = np.random.uniform(-1.0, 1.0, size=[36, self.noise_dim])
        test_image -= 0.5
        test_image /= 0.5
        ###
        test_elev_code *= 0.5
        test_elev_code += 0.5
        test_azim_code *= 0.5
        test_azim_code += 0.5
        ###
        plot_image(test_image, color=self.color)

        valid = np.ones([self.batch_size, 1])
        fake = np.zeros([self.batch_size, 1])
                            
        valid_fake = np.concatenate((valid, fake))
        epochs = 120
        train_steps = self.train_steps * epochs

        fake_pc_codes = np.load(self.pc_codes_filename)
        fake_pc_codes_len = len(fake_pc_codes)
        print("Loaded pc codes", self.pc_codes_filename, " with len: ", fake_pc_codes_len)
        print("fake_pc_codes min: ", np.amin(fake_pc_codes), "fake_pc_codes max: ", np.amax(fake_pc_codes))
        print("test_pc_code min: ", np.amin(test_pc_code), " test_pc_code max: ", np.amax(test_pc_code))
        print("test_elev_code min: ", np.amin(test_elev_code), " test_elev_code max: ", np.amax(test_elev_code))
        print("test_azim_code min: ", np.amin(test_azim_code), " test_azim_code max: ", np.amax(test_azim_code))
        print("batch_size: ", self.batch_size, " pc_code_dim: ", self.pc_code_dim)
        print("Color images: ", self.color)

        for step in range(train_steps):
            real_image, real_pc, real_elev_code, real_azim_code = self.train_source.next_batch()
            real_image -= 0.5
            real_image /= 0.5
            # pc is [-0.5, 0.5]
            real_pc = real_pc / 0.5
            real_pc_code = self.ptcloud_ae.encoder.predict(real_pc)

            rand_indexes = np.random.randint(0, fake_pc_codes_len, size=self.batch_size) 
            fake_pc_code = fake_pc_codes[rand_indexes]

            pc_code = np.concatenate((real_pc_code, fake_pc_code))

            ###
            # fake_view_code = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.view_dim])
            real_elev_code *= 0.5
            real_elev_code += 0.5
            fake_elev_code = np.random.uniform(0.0, 1.0, size=[self.batch_size, 1])
            real_azim_code *= 0.5
            real_azim_code += 0.5
            fake_azim_code = np.random.uniform(0.0, 1.0, size=[self.batch_size, 1])
            ###

            elev_code = np.concatenate((real_elev_code, fake_elev_code))
            azim_code = np.concatenate((real_azim_code, fake_azim_code))

            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.noise_dim])
            fake_image = self.generator.predict([noise, fake_pc_code, fake_elev_code, fake_azim_code])
            x = np.concatenate((real_image, fake_image))
            metrics  = self.discriminator.train_on_batch(x, [valid_fake, pc_code, elev_code, azim_code])
            pcent = step * 100.0 / train_steps
            fmt = "%02.4f%%/%06d:[loss:%02.6f d:%02.6f pc:%02.6f elev:%02.6f azim:%02.6f]" 
            log = fmt % (pcent, step, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])

            rand_indexes = np.random.randint(0, fake_pc_codes_len, size=self.batch_size) 
            fake_pc_code = fake_pc_codes[rand_indexes]

            ###
            # fake_view_code = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.view_dim])
            fake_elev_code = np.random.uniform(0.0, 1.0, size=[self.batch_size, 1])
            fake_azim_code = np.random.uniform(0.0, 1.0, size=[self.batch_size, 1])
            ###

            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.noise_dim])

            metrics  = self.adversarial.train_on_batch([noise, fake_pc_code, fake_elev_code, fake_azim_code],
                                                       [valid, fake_pc_code, fake_elev_code, fake_azim_code])
            fmt = "%s [loss:%02.6f a:%02.6f pc:%02.6f elev:%02.6f azim:%02.6f]" 
            log = fmt % (log, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])

            elapsed_time = datetime.datetime.now() - start_time
            log = "%s [time: %s]" % (log, elapsed_time)
            print(log)
            if (step + 1) % plot_interval == 0 or step == 0:
                # plot generator images on a periodic basis
                show = False
                plot_images(self.generator,
                            noise=noise_,
                            pc_code=test_pc_code,
                            elev_code=test_elev_code,
                            azim_code=test_azim_code,
                            color=self.color,
                            show=show,
                            step=(step + 1))

            if (step + 1) % save_interval == 0 or step == 0:
                # save weights on a periodic basis

                prefix = self.category + "-gen"
                if self.color:
                    prefix += "-color"
                else:
                    prefix += "-gray"
                if self.gen_spectral_normalization:
                    prefix += "-sn"
                prefix += "-" + str(self.pc_code_dim)
                fname = os.path.join("weights", prefix + ".h5")
                self.generator_single.save_weights(fname)
                prefix = self.category + "-dis"
                if self.color:
                    prefix += "-color"
                else:
                    prefix += "-gray"
                if self.gen_spectral_normalization:
                    prefix += "-sn"
                prefix += "-" + str(self.pc_code_dim)
                fname = os.path.join("weights", prefix + ".h5")
                self.discriminator_single.save_weights(fname)


    def azim_loss(self, y_true, y_pred):
        rad = 2. * np.pi
        rad *= (y_true - y_pred)
        return K.mean(K.abs(tf.atan2(K.sin(rad), K.cos(rad))), axis=-1)

    def elev_loss(self, y_true, y_pred):
        # rad = 2. * np.pi * 80. /360.
        rad = 0.4444444444444444 * np.pi
        rad *= (y_true - y_pred)
        return K.mean(K.abs(tf.atan2(K.sin(rad), K.cos(rad))), axis=-1)

    def build_gan(self):
        # set if generator is going to use spectral norm
        image, pc, elev, azim = self.train_source.next_batch()
        elev_code = Input(shape=(1,), name='elev_code')
        azim_code = Input(shape=(1,), name='azim_code')
        pc_code = Input(shape=(self.pc_code_dim,), name='pc_code')
        noise_code = Input(shape=(self.noise_dim,), name='noise_code')
        model_name = "pc2pix"
        image_size = image.shape[1]
        if self.color:
            input_shape = (image_size, image_size, 3)
        else:
            input_shape = (image_size, image_size, 1)

        inputs = Input(shape=input_shape, name='image_input')
        if self.gen_spectral_normalization:
            optimizer = Adam(lr=4e-4, beta_1=0.0, beta_2=0.9)
        else:
            optimizer = Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)

        # build discriminator
        # by default, discriminator uses SN
        if self.gpus <= 1:
            self.discriminator= model.discriminator(input_shape,
                                                    pc_code_dim=self.pc_code_dim)
            if self.dw is not None:
                print("loading discriminator weights: ", self.dw)
                self.discriminator.load_weights(self.dw)
            self.discriminator_single = self.discriminator
        else:
            with tf.device("/cpu:0"):
                self.discriminator_single = model.discriminator(input_shape,
                                                                pc_code_dim=self.pc_code_dim)
                if self.dw is not None:
                    print("loading discriminator weights: ", self.dw)
                    self.discriminator_single.load_weights(self.dw)

            self.discriminator = multi_gpu_model(self.discriminator_single, gpus=self.gpus)
	
        loss = ['binary_crossentropy', 'mae', self.elev_loss, self.azim_loss]
        loss_weights = [1., 300., 10., 50.]
        self.discriminator.compile(loss=loss,
                                   loss_weights=loss_weights,
                                   optimizer=optimizer)
        self.discriminator_single.summary()
        path = os.path.join(self.model_dir, "discriminator.png")
        plot_model(self.discriminator_single, to_file=path, show_shapes=True)

        # build generator
        # try SN to see if mode collapse is avoided
        if self.gpus <= 1:
            self.generator = model.generator(input_shape,
                                             noise_code=noise_code,
                                             pc_code=pc_code,
                                             elev_code=elev_code,
                                             azim_code=azim_code,
                                             spectral_normalization=self.gen_spectral_normalization,
                                             color=self.color)
            if self.gw is not None:
                print("loading generator weights: ", self.gw)
                self.generator.load_weights(self.gw)
            self.generator_single = self.generator
        else:
            with tf.device("/cpu:0"):
                self.generator_single = model.generator(input_shape,
                                                        noise_code=noise_code,
                                                        pc_code=pc_code,
                                                        elev_code=elev_code,
                                                        azim_code=azim_code,
                                                        spectral_normalization=self.gen_spectral_normalization,
                                                        color=self.color)
                if self.gw is not None:
                    print("loading generator weights: ", self.gw)
                    self.generator_single.load_weights(self.gw)

            self.generator = multi_gpu_model(self.generator_single, gpus=self.gpus)

        self.generator_single.summary()
        path = os.path.join(self.model_dir, "generator.png")
        plot_model(self.generator_single, to_file=path, show_shapes=True)
        
        self.discriminator.trainable = False
        if self.gen_spectral_normalization:
            optimizer = Adam(lr=1e-4, beta_1=0.0, beta_2=0.9)
        else:
            optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)

        if self.gpus <= 1:
            self.adversarial= Model([noise_code, pc_code, elev_code, azim_code],
                                    self.discriminator(self.generator([noise_code, pc_code, elev_code, azim_code])),
                                    name=model_name)
            self.adversarial_single = self.adversarial
        else:
            with tf.device("/cpu:0"):
                self.adversarial_single = Model([noise_code, pc_code, elev_code, azim_code],
                                                self.discriminator(self.generator([noise_code, pc_code, elev_code, azim_code])),
                                                name=model_name)
            self.adversarial = multi_gpu_model(self.adversarial_single, gpus=self.gpus)

        self.adversarial.compile(loss=loss,
                                 loss_weights=loss_weights,
                                 optimizer=optimizer)
        self.adversarial_single.summary()
        path = os.path.join(self.model_dir, "adversarial.png")
        plot_model(self.adversarial_single, to_file=path, show_shapes=True)

        print("Using split file: ", self.split_file)
        print("1 epoch datalen: ", self.epoch_datalen)
        print("1 epoch train steps: ", self.train_steps)
        print("Using pc codes: ", self.pc_codes_filename)


    def stop_sources(self):
        self.train_source.close()
        self.test_source.close()


    def __del__(self):
        self.stop_sources()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator model trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Load discriminator model trained weights"
    parser.add_argument("-d", "--discriminator", help=help_)
    help_ = "Load h5 ptcloud_ae model trained ae weights"
    parser.add_argument("-w", "--ptcloud_ae_weights", help=help_)
    help_ = "Train pc2pix"
    parser.add_argument("-t", "--train", default=False, action='store_true', help=help_)
    help_ = "Use grayscale images 224x224pix"
    parser.add_argument("--gray", default=False, action='store_true', help=help_)
    help_ = "Point cloud code dim"
    parser.add_argument("-p", "--pc_code_dim", type=int, default=32, help=help_)
    help_ = "Batch size"
    parser.add_argument("-b", "--batch_size", type=int, default=64, help=help_)
    help_ = "Shapnet category or class (chair, airplane, etc)"
    parser.add_argument("-c", "--category", default='all', help=help_)
    help_ = "Number of GPUs (default is 1)"
    parser.add_argument("--gpus", type=int, default=1, help=help_)
    args = parser.parse_args()

    gw = None
    dw = None

    if args.discriminator:
        dw = args.discriminator

    if args.generator:
        gw = args.generator

    ptcloud_ae = PtCloudStackedAE(latent_dim=args.pc_code_dim,
                                  evaluate=True,
                                  category=args.category)
    ptcloud_ae.stop_sources()

    if args.ptcloud_ae_weights:
        print("Loading point cloud ae weights: ", args.ptcloud_ae_weights)
        ptcloud_ae.ae.load_weights(args.ptcloud_ae_weights)
    else:
        print("Trained point cloud ae required to pc2pix")
        exit(0)

    pc2pix = PC2Pix(ptcloud_ae=ptcloud_ae,
                    gw=gw,
                    dw=dw,
                    batch_size=args.batch_size,
                    pc_code_dim=args.pc_code_dim,
                    category=args.category,
                    color=(not args.gray),
                    gpus=args.gpus)
    if args.train:
        pc2pix.train_gan()
    pc2pix.stop_sources()
    del pc2pix

