import numpy as np
import os
from os import path
import pickle
import sys
import matplotlib.image as img
import matplotlib.pyplot as plt
from os.path import join
from scipy import misc
import argparse
from random import shuffle
from scipy import io

from shapenet import ShapeNet

from config import SHAPENET_IM, SHAPENET_VOX
import tensorflow as tf
from general_utils import plot_3d_point_cloud

def load_subdirs(path):
    subdirs = list_files(path)
    return subdirs

class DataSource():
    def __init__(self,
                 batch_size=16,
                 smids='train',
                 mid=None,
                 items=['im', 'pc'],
                 split_file='data/all_exp.json',
                 nepochs=None):
        vox_dir = SHAPENET_VOX[32]
        im_dir = SHAPENET_IM
        self.coord = tf.train.Coordinator()
        self.dset = ShapeNet(im_dir=im_dir, vox_dir=vox_dir, split_file=split_file)
        mids = self.dset.get_smids(smids)
        if mid is not None:
            mids[0][1] = mid
            mids = mids[0:1]
        # print(mids[0])
        # print(mids.shape)
        print("Mids", len(mids))
        self.items = items
        self.batch_size = batch_size
        self.dset.init_queue(mids,
                             1,  # maybe this is the number of images per mid per batch
                             self.items,
                             self.coord,
                             qsize=8,
                             nthreads=8,
                             nepochs=nepochs)

    def close(self):
        self.dset.close_queue()
        self.coord.join()


    def next_batch(self):
        try:
            batch_data = self.dset.next_batch(self.items,
                                              self.batch_size)
            if batch_data is None:
                return None

            ret = []
            if 'im' in self.items:
                img = batch_data['im']
                s = img.shape
                img = np.reshape(img, (-1, s[2], s[3], s[4]))
                ret.append(img)

            if 'im_128' in self.items:
                img = batch_data['im_128']
                s = img.shape
                img = np.reshape(img, (-1, s[2], s[3], s[4]))
                ret.append(img)

            if 'gray' in self.items:
                img = batch_data['gray']
                s = img.shape
                img = np.reshape(img, (-1, s[2], s[3], 1))
                ret.append(img)

            if 'gray_128' in self.items:
                img = batch_data['gray_128']
                s = img.shape
                img = np.reshape(img, (-1, s[2], s[3], 1))
                ret.append(img)

            if 'pc' in self.items:
                pc = batch_data['pc']
                s = pc.shape
                pc = np.reshape(pc, (-1, s[2], s[3]))
                ret.append(pc)

            if 'elev' in self.items:
                elev = batch_data['elev']
                s = elev.shape
                elev = np.reshape(elev, (-1, s[1]*s[2]))
                ret.append(elev)

            if 'azim' in self.items:
                azim = batch_data['azim']
                s = azim.shape
                azim = np.reshape(azim, (-1, s[1]*s[2]))
                ret.append(azim)

            if 'view' in self.items:
                # vox = batch_data['vol']
                # s = vox.shape
                # vox = np.reshape(vox, (-1, s[1], s[2], s[3]))
                # return img, pc, vox
                view = batch_data['view']
                s = view.shape
                view = np.reshape(view, (-1, s[1]*s[2]))
                ret.append(view)

            return tuple(ret) 
        except Exception as e:
            self.dset.close_queue(e)
            print("Exception: ", e)
        # finally:
            #print("finally")
            # self.coord.join()


    def save_vox(self, vox, path="model.mat"):
        dic = {}
        dic["vox"] = vox
        io.savemat(path, dic)


    def save_img(self, img, option=None, path="image.png"):
        if option is not None:
            plt.imsave(path, img, cmap=option)
        else:
            plt.imsave(path, img)
    

def list_files(dir_):
    files = []
    files.extend([f for f in sorted(os.listdir(dir_)) ])
    return files


def load_samples():
    im_dir = SHAPENET_IM
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, "data")
    split_file = os.path.join(path, "splits.json")
    dset = ShapeNet(im_dir=im_dir, split_file=split_file, rng_seed=1)

    # smids = 'test' mids='02691156'
    x = dset.get_smids('test')
    x = np.array(x)
    print(x[0][0])   # sid
    print(x[0][1])   # mid
    print(x[1][0])
    print(x[1][1])
    print(x.shape)

    return
    train = np.array(x['train'])
    print(train.shape)
    val = np.array(x['val'])
    print(val.shape)
    test = np.array(x['test'])
    print(test.shape)

    print(val)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    load_samples()
    exit(0)
    
    dsource = DataSource()
    ims, pcs = dsource.next_batch()
    for i in range(len(ims)):
        im = ims[i]
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, "tmp")
        f = "im-%d.png" % i
        path = os.path.join(path, f)
        dsource.save_img(im, path)
        reconstruction = pcs[i]
        plot_3d_point_cloud(reconstruction[:, 0],
                            reconstruction[:, 1],
                            reconstruction[:, 2], in_u_sphere=True);
    print(ims.shape)
    print(pcs.shape)
    dsource.close()
