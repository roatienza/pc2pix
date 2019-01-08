from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from general_utils import plot_3d_point_cloud
# from evaluate import get_iou, get_emd
import sys
sys.path.append("external")
from in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder
from general_utils import plot_3d_point_cloud
# from structural_losses.tf_nndistance import nn_distance
#from structural_losses.tf_approxmatch import approx_match, match_cost
# from tf_ops.emd import tf_auctionmatch
# from tf_ops.sampling import tf_sampling
# from tf_ops.CD import tf_nndistance


def save_weights(model, model_name, weights_dir, latent_dim, prefix='model'):
    weights_dir = os.path.join(os.getcwd(), weights_dir)
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)
    filename = '%s-%s-weights-%d.h5' % (prefix, model_name, latent_dim)
    path = os.path.join(weights_dir, filename)
    model.save_weights(path)

    # filename = '%s-%s-model-%d.h5' % (prefix, model_name, latent_dim)
    # path = os.path.join(weights_dir, filename)
    # model.save(path)


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def save_images(images, datasource, gray=False, folder="tmp"):
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, folder)
    i = 0
    for image in images:
        filename = "image-%d.png" % i
        image_path = os.path.join(path, filename)
        if gray:
            shape = image.shape 
            plt.imsave(image_path, np.reshape(image, (shape[0], shape[0])), cmap='gray')
        else:
            datasource.save_img(image, image_path)
        i += 1


def plot_vox(vox, title="vox"):
    fig = plt.figure()
    plt.title(title)
    ax = fig.gca(projection='3d')
    ax.voxels(vox, edgecolor='b')
    ax.view_init(azim=240, elev=10)
    plt.show()
    # plt.close('all')


def ptcloud_to_vox(ptcloud, dim=32):
    vox = np.zeros([dim, dim, dim])
    for idx in ptcloud:
        vox[tuple(idx)] = 1

    return vox

def get_all_iou(posterior_net, ptcloud_ae):
    im, pc, vol = posterior_net.test_source.next_batch()


def plot_image2ptcloud_results(image2ptcloud):
    im, pc, vol = image2ptcloud.test_source.next_batch()
    print("pc: ", pc.shape)
    save_images(im, posterior_net.test_source)
    
    pre = image2ptcloud.predict(im)
    pre *= 0.5
    iou = 0.0
    for i in range(len(latent)):
        grd = pc[i]
        print("Ground: ", i)
        plot_3d_point_cloud(grd[:, 0], 
                            grd[:, 1], 
                            grd[:, 2], title='point cloud', in_u_sphere=True);

        print("Prediction: ", i)
        plot_3d_point_cloud(pre[i][:, 0], 
                            pre[i][:, 1], 
                            pre[i][:, 2], title='point cloud', in_u_sphere=True);


def get_chamfer(gt, pred, batch_size=32, version=1):
    if version==0:
        p1top2, _, p2top1, _ = nn_distance(pred, gt)
        cd = K.mean(p1top2, axis=1) + K.mean(p2top1, axis=1)
        return K.eval(cd)

    p1top2 , _, p2top1, _ = tf_nndistance.nn_distance(gt, pred)
    #p1top2 is for each element in gt, the cloest distance to this element
    cd = p1top2 #  + p2top1
    cd = K.mean(cd, axis=1)
    return K.eval(cd)


def get_emd(gt, pred, batch_size=32, version=1):
     matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
     matched_out = tf_sampling.gather_point(gt, matchl_out)
     emd = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
     emd = tf.reduce_mean(emd, axis=1, keepdims=True)
     emd = K.eval(emd)
     return emd


def eval_posterior_net_results(posterior_net, ptcloud_ae, emd_metric=True, version=1):
    metric = None
    batch = 0
    cost = get_emd
    if not emd_metric:
        cost = get_chamfer
    while True:
        im, pc, view = posterior_net.test_source.next_batch()
        if im is None:
            break

        batch += posterior_net.batch_size
        latent = posterior_net.posterior_net.predict(im)
        pred = ptcloud_ae.decoder.predict(latent)
        # ground truth is [-0.5, 0.5]
        # prediction is [-1.0, 1.0]
        # rescale
        pred *= 0.5
        metric_ = cost(pc, pred, version=version)
        if metric is None:
            metric = metric_
        else:
            metric = np.append(metric, metric_, axis=0)
        if emd_metric:
            print("EMD", metric.shape, ": ", metric.mean())
        else:
            print("Sqrt Chamfer ver", str(version), metric.shape[0], ": ", np.sqrt(metric).mean())

    if emd_metric:
        print("Mean EMD: ", metric.mean())
    else:
        print("Mean Sqrt Chamfer: ", np.sqrt(metric).mean())


def plot_posterior_net_results(posterior_net, ptcloud_ae=None, image_ae=None, im2pc=True):
    im, pc, view = posterior_net.test_source.next_batch()
    print("pc: ", pc.shape)
    # save_images(im, posterior_net.test_source)
   
    if im2pc:
        latent = posterior_net.posterior_net.predict(im)
        pre = ptcloud_ae.decoder.predict(latent)
        pre *= 0.5
        cost = get_chamfer(pc, pre)
        cost = np.sqrt(cost)
        print("Batch chamfer: ", cost.mean())
        for i in range(len(latent)):
            grd = pc[i]
            print("Ground: ", i)
            plot_3d_point_cloud(grd[:, 0], 
                                grd[:, 1], 
                                grd[:, 2], title='point cloud', in_u_sphere=True);

            print("Prediction: ", i)
            print("Chamfer: ", cost[i])
            plot_3d_point_cloud(pre[i][:, 0], 
                                pre[i][:, 1], 
                                pre[i][:, 2], title='point cloud', in_u_sphere=True);
    else:
        latent, v = posterior_net.posterior_net.predict([pc, view])
        pre  = image_ae.decoder.predict(latent)
        for i in range(len(latent)):
            grd = pc[i]
            print("Prediction: ", i)
            print("View ground: ", view[i], " View pred: ", v[i])
            img = pre[i]
            s = img.shape
            img = np.reshape(img, (s[0], s[0]))
            save_image(img, i, folder="predictions")
            plot_3d_point_cloud(grd[:, 0], 
                                grd[:, 1], 
                                grd[:, 2], title='point cloud', in_u_sphere=True);


def save_image(image, index, folder="tmp"):
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, folder)
    filename = "image-%d.png" % index
    image_path = os.path.join(path, filename)
    print(image_path)
    plt.imsave(image_path, image, cmap='gray')
    # datasource.save_img(image, image_path, option='gray')


def plot_decoded_ptcloud(ae, test_source):

    im, pc = test_source.next_batch()
    data_dir = 'ptcloud_out'
    save_dir = os.path.join(os.getcwd(), data_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i in range(len(pc)):
        test = pc[i]
        plot_3d_point_cloud(test[:, 0], 
                            test[:, 1], 
                            test[:, 2], in_u_sphere=True);

        test = pc[i] / 0.5
        shape = (1,) + test.shape
        test = np.reshape(test, shape)
        reconstruction = 0.5 * ae.predict(test)
        plot_3d_point_cloud(reconstruction[0][:, 0], 
                            reconstruction[0][:, 1], 
                            reconstruction[0][:, 2], in_u_sphere=True);


def plot_ae_results(models,
                    test_data,
                    gray=True,
                    model_name="ae_test"):

    encoder, decoder, ae = models
    data_dir = 'predictions'
    save_dir = os.path.join(os.getcwd(), data_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    outputs = ae.predict(test_data)
    nitems = outputs.shape[0]

    for i in range(nitems):
        filename = 'pred-ae-%d.png' % i
        path = os.path.join(data_dir, filename)
        if gray:
            shape = outputs[i].shape 
            plt.imsave(path, np.reshape(outputs[i], (shape[0], shape[0])), cmap='gray')
        else:
            plt.imsave(path, outputs[i])

        filename = 'grnd-ae-%d.png' % i
        if gray:
            plt.imsave(path, np.reshape(test_data[i], (shape[0], shape[0])), cmap='gray')
        else:
            path = os.path.join(data_dir, filename)
        plt.imsave(path, test_data[i])


def plot_vae_results(models, test_data, latent_dim=1024, gray=False, data_dir="predictions"):

    encoder, decoder, vae = models
    save_dir = os.path.join(os.getcwd(), data_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    outputs = vae.predict(test_data)
    nitems = outputs.shape[0]

    for i in range(nitems):
        filename = 'pred-vae-%d.png' % i
        path = os.path.join(data_dir, filename)
        # plt.imsave(path, outputs[i])
        image = outputs[i]
        shape = image.shape 
        plt.imsave(path, np.reshape(image, (shape[0], shape[0])), cmap='gray')

        filename = 'grnd-vae-%d.png' % i
        path = os.path.join(data_dir, filename)
        # plt.imsave(path, test_data[i])
        image = test_data[i]
        plt.imsave(path, np.reshape(image, (shape[0], shape[0])), cmap='gray')

        lshape = (1, latent_dim)
        z_sample = np.random.uniform(-1, 1, lshape)
        reco = decoder.predict(z_sample)
        filename = 'reco-vae-%d.png' % i
        path = os.path.join(data_dir, filename)
        # plt.imsave(path, reco[0])
        image = reco[0]
        plt.imsave(path, np.reshape(image, (shape[0], shape[0])), cmap='gray')
