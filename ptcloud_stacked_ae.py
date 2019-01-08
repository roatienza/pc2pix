'''Point cloud stacked autoencoder.

To run evaluation of autoencoder:
python3 ptcloud_stacked_ae.py --weights=model_weights/ptcloud/all-pt-cloud-stacked-ae-emd-5-ae-weights-512.h5 -l=512 -e -k=5

To run training:
if emd loss
    python3 ptcloud_stacked_ae.py --weights=model_weights/ptcloud/all-pt-cloud-stacked-ae-emd-5-ae-weights-512.h5 -l=512 -t -k=5
if chamfer loss
    python3 ptcloud_stacked_ae.py --weights=model_weights/all-pt-cloud-stacked-ae-chamfer-5-ae-weights-512.h5 -l=512 -t -k=5 --chamfer

'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input, Activation
from keras.layers import Conv1D, Flatten 
from keras.layers import Reshape, UpSampling1D, BatchNormalization, MaxPooling1D
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import RMSprop
from keras.layers.merge import concatenate
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import datetime
import sys

sys.path.append("external")
from general_utils import plot_3d_point_cloud
# from tf_ops.emd import tf_auctionmatch
# from tf_ops.sampling import tf_sampling
# from tf_ops.CD import tf_nndistance
#from structural_losses import tf_nndistance

from data import DataSource
from model_utils import save_weights, save_images
from model_utils import plot_decoded_ptcloud


class PtCloudStackedAE():
    def __init__(self,
                 latent_dim=32,
                 kernel_size=5,
                 lr=1e-4,
                 category="all",
                 evaluate=False,
                 emd=True):

        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = 32
        self.evaluate = evaluate
        self.emd = emd
        self.inputs = None
        self.encoder = None
        self.decoder = None
        self.ae = None
        self.z_log_var = None
        self.z_mean = None
        self.z = None
        self.kernel_size = kernel_size
        batch_size = 32
        self.model_dir = "saved_models"
        os.makedirs(self.model_dir, exist_ok=True) 
        self.category = category
        if category == 'all':
            path = 'all_exp_norm.json'
        else:
            path = category + '_exp.json'
        split_file = os.path.join('data', path)
        print("Using train split file: ", split_file)

        self.train_source = DataSource(batch_size=batch_size, split_file=split_file)
        self.test_source = DataSource(batch_size=batch_size, smids='test', nepochs=20, split_file=split_file)
        shapenet = self.train_source.dset
        self.epoch_datalen = len(shapenet.get_smids('train'))*shapenet.num_renders
        self.train_steps = len(shapenet.get_smids('train'))*shapenet.num_renders // self.batch_size
        _, pc = self.train_source.next_batch()
        self.input_shape = pc[0].shape
        self.build_ae()
        
    def encoder_layer(self, x, filters, strides=1, dilation_rate=1):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(filters=filters,
                   kernel_size=self.kernel_size,
                   strides=strides,
                   dilation_rate=dilation_rate,
                   padding='same')(x)
        return x

    def compression_layer(self, x, y, maxpool=True):
        if maxpool:
            y = MaxPooling1D()(y)
        x = concatenate([x, y])

        y = Conv1D(filters=64,
                   kernel_size=1,
                   activation='relu',
                   padding='same')(x)
        return x, y
        
    def build_encoder(self, filters=64, activation='linear'):

        self.inputs = Input(shape=self.input_shape, name='encoder_input')
        x = self.inputs
        y = self.inputs
        strides = 2
        maxpool = True
        x1 = self.encoder_layer(x, filters, strides=1, dilation_rate=1)
        x2 = self.encoder_layer(x, filters, strides=1, dilation_rate=2)
        x4 = self.encoder_layer(x, filters, strides=1, dilation_rate=4)
        x8 = self.encoder_layer(x, filters, strides=1, dilation_rate=8)
        x = concatenate([x1, x2, x4, x8])
        x, y = self.compression_layer(x, y, maxpool=False)

        x = self.encoder_layer(x, 128, strides=2, dilation_rate=1)

        x1 = self.encoder_layer(x, filters, strides=1, dilation_rate=1)
        x2 = self.encoder_layer(x, filters, strides=1, dilation_rate=2)
        x4 = self.encoder_layer(x, filters, strides=1, dilation_rate=4)
        x8 = self.encoder_layer(x, filters, strides=1, dilation_rate=8)
        x = concatenate([x1, x2, x4, x8])
        x, y = self.compression_layer(x, y, maxpool=True)

        x = self.encoder_layer(x, 128, strides=2, dilation_rate=1)

        x1 = self.encoder_layer(x, filters, strides=1, dilation_rate=1)
        x2 = self.encoder_layer(x, filters, strides=1, dilation_rate=2)
        x4 = self.encoder_layer(x, filters, strides=1, dilation_rate=4)
        x8 = self.encoder_layer(x, filters, strides=1, dilation_rate=8)
        x = concatenate([x1, x2, x4, x8])
        x, y = self.compression_layer(x, y, maxpool=True)

        x = self.encoder_layer(x, 128, strides=2, dilation_rate=1)

        x1 = self.encoder_layer(x, filters, strides=1, dilation_rate=1)
        x2 = self.encoder_layer(x, filters, strides=1, dilation_rate=2)
        x4 = self.encoder_layer(x, filters, strides=1, dilation_rate=4)
        x8 = self.encoder_layer(x, filters, strides=1, dilation_rate=8)
        x = concatenate([x1, x2, x4, x8])
        x, y = self.compression_layer(x, y, maxpool=True)

        x = self.encoder_layer(x, 32)
        shape = K.int_shape(x)

        x = Flatten()(x)
        # x = Dense(128, activation='relu')(x)
        # experimental tanh activation, revert to none or linear if needed
        outputs = Dense(self.latent_dim, activation=activation, name='ae_encoder_out')(x)
        path = os.path.join(self.model_dir, "ae_encoder.png")
        self.encoder = Model(self.inputs, outputs, name='ae_encoder')

        self.encoder.summary()
        plot_model(self.encoder, to_file=path, show_shapes=True)

        return shape, filters


    def build_decoder_mlp(self, dim=1024):

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        x = latent_inputs
        x = Dense(dim, activation='relu')(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(np.prod(self.input_shape), activation='tanh')(x)
        outputs = Reshape(self.input_shape)(x)

        path = os.path.join(self.model_dir, "decoder_mlp.png")
        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()
        plot_model(self.decoder, to_file=path, show_shapes=True)


    def build_decoder(self, filters, shape):

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        pt_cloud_shape = (shape[1], shape[2])
        dim = shape[1] * shape[2]
        x = Dense(128, activation='relu')(latent_inputs)
        x = Dense(dim, activation='relu')(x)
        x = Reshape(pt_cloud_shape)(x)

        for i in range(4):
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv1D(filters=filters,
                       kernel_size=self.kernel_size,
                       padding='same')(x)
            x = UpSampling1D()(x)
            filters //= 2

        outputs = Conv1D(filters=3,
                         kernel_size=self.kernel_size,
                         activation='tanh',
                         padding='same',
                         name='decoder_output')(x)

        path = os.path.join(self.model_dir, "decoder.png")
        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()
        plot_model(self.decoder, to_file=path, show_shapes=True)


    def loss(self, gt, pred):
        from tf_ops.emd import tf_auctionmatch
        from tf_ops.sampling import tf_sampling
        # from tf_ops.CD import tf_nndistance
        from structural_losses import tf_nndistance
        if self.emd:
            matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
            matched_out = tf_sampling.gather_point(gt, matchl_out)
            emd_loss = tf.reshape((pred - matched_out) ** 2, shape=(self.batch_size, -1))
            emd_loss = tf.reduce_mean(emd_loss, axis=1, keepdims=True)
            return emd_loss
        else:
            p1top2 , _, p2top1, _ = tf_nndistance.nn_distance(gt, pred)
            #p1top2 is for each element in gt, the cloest distance to this element
            cd_loss = p1top2 + p2top1
            cd_loss = K.mean(cd_loss)
            return cd_loss


    def build_ae(self):
        shape, filters = self.build_encoder()
        decoder = self.build_decoder_mlp()

        outputs = self.decoder(self.encoder(self.inputs))
        self.ae = Model(self.inputs, outputs, name='ae')

        self.ae.summary()
        #if not self.evaluate:
        #    self.ae.add_loss(self.loss)
        optimizer = RMSprop(lr=self.lr)
        if not self.evaluate:
            self.ae.compile(optimizer=optimizer, loss=self.loss)
        path = os.path.join(self.model_dir, "ae.png")
        plot_model(self.ae, to_file=path, show_shapes=True)
        print("Learning rate: ", self.lr)


    def train_ae(self):
        epochs = 200
        train_steps = self.train_steps * epochs
        save_interval = 500
        print_interval = 100
        start_time = datetime.datetime.now()
        loss = 0.0
        epochs = 400
        train_steps = self.train_steps * epochs

        for step in range(train_steps):
            _, pc = self.train_source.next_batch()
            pc = pc / 0.5
            metrics = self.ae.train_on_batch(x=pc, y=pc)
            loss += metrics

            if (step + 1) % print_interval == 0:
                elapsed_time = datetime.datetime.now() - start_time
                loss /= print_interval
                pcent = step * 100.0 / train_steps
                fmt = "%02.4f%%/%06d:[loss:%02.6f time:%s]" 
                log = fmt % (pcent, step + 1, loss, elapsed_time)
                # log = "%d: [loss: %0.6f] [time: %s]" % (step + 1, loss, elapsed_time)
                print(log)
                loss = 0.0


            if (step + 1) % save_interval == 0:
                prefix = self.category + "-" + "pt-cloud-stacked-ae"
                if self.emd:
                    prefix += "-emd"
                else:
                    prefix += "-chamfer"
                prefix += "-" + str(self.kernel_size)
                weights_dir = "weights"
                save_weights(self.encoder,
                             "encoder",
                             weights_dir,
                             self.latent_dim,
                             prefix=prefix)
                save_weights(self.decoder,
                             "decoder",
                             weights_dir,
                             self.latent_dim,
                             prefix=prefix)
                save_weights(self.ae,
                             "ae",
                             weights_dir,
                             self.latent_dim,
                             prefix=prefix)

    def stop_sources(self):
        self.train_source.close()
        self.test_source.close()

    def __del__(self):
        self.stop_sources()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Train the autoencoder"
    parser.add_argument("-t", "--train", action='store_true', help=help_)
    help_ = "Latent dim"
    parser.add_argument("-l", "--latent_dim", default=32, type=int, help=help_)
    help_ = "Kernel size"
    parser.add_argument("-k", "--kernel_size", default=1, type=int, help=help_)
    help_ = "Learning rate"
    parser.add_argument("-r", "--lr", default=1e-4, type=float, help=help_)
    help_ = "Evaluate autoencoder"
    parser.add_argument("-e", "--evaluate", default=False, action='store_true', help=help_)
    help_ = "Use Chamder distance loss"
    parser.add_argument("--chamfer", default=False, action='store_true', help=help_)
    help_ = "Shapnet category or class (chair, airplane, etc)"
    parser.add_argument("-a", "--category", default='all', help=help_)
    args = parser.parse_args()

    ptcloud_ae = PtCloudStackedAE(latent_dim=args.latent_dim,
                                  kernel_size=args.kernel_size,
                                  lr=args.lr,
                                  category=args.category,
                                  evaluate=args.evaluate,
                                  emd=not args.chamfer)

    if args.weights:
        print("Loading ", args.weights)
        ptcloud_ae.ae.load_weights(args.weights)
        # save_test(im, test_source)
        if args.evaluate:
            plot_decoded_ptcloud(ptcloud_ae.ae, ptcloud_ae.test_source)

    print("latent dim : ",
          args.latent_dim, " lr: ",
          args.lr,
          " kernel size: ",
          args.kernel_size,
          " category: ",
          args.category)
    if args.train:
        ptcloud_ae.train_ae()

    ptcloud_ae.stop_sources()
