pc2pix code implementation in Keras(2.2.3) and Tensorflow (1.11)
surface reconstruction requires blender and meshlab (meshlab server)

Environment tested: Ubunutu Linux 16.04LTS, GPUs tested: 1060, 1080Ti and V100

Datasets (ShapeNet point cloud and rendered images) can be downloaded here (1G + 28G):
$ cd data
Download shapenet point cloud dataset from https://bit.ly/2RFzBeG
$ tar xzvf  shape_net_core_uniform_samples_2048.tgz
Download shapenet render dataset from https://bit.ly/2RTa54Z 
$ tar xzvf shapenet_release.tgz
$ cd ..


Pre-compiled latent codes (to speed up point cloud encoder prediction)
Note: Needed only if pc2pix will be trained
$ Download pc codes from https://goo.gl/JBL4sU or https://bit.ly/2AUra4N
$ tar jxvf pc_codes.tar.bz2

Model weights of pc2pix and pt cloud autoencoder (1G)
$ Download model weights from https://goo.gl/3vMuxY or https://bit.ly/2CzFAae
$ tar jxvf model_weights.tar.bz2


Evaluation code can be found in evaluation folder. All rendered images will be at evaluation/data
$ cd evaluation

1) For simplicity, assume chair dataset is used:

2) To perform surface surface reconstrucion from point clouds:
$ python3 surface_reconstruction.py

3) To render the surface reconstructed objects:
$ python3 render_reconstruction.py

4) To render point cloud using pc2pix:
$ python3 render_by_pc2pix.py --ptcloud_ae_weights=../model_weights/ptcloud/chair-pt-cloud-stacked-ae-chamfer-5-ae-weights-32.h5 --generator=../model_weights/pc2pix/chair-gen-color.h5 --discriminator=../model_weights/pc2pix/chair-dis-color.h5 -c --category="chair"

5) To calculate FID scores:
$ python3 get_fid.py

6) To calculate SSIM components:
$ python3 get_ssim_components.py

7) To get class similarity:
$ python3 get_class_confidence.py

$ cd ..


Compile CUDA code for Chamfer Distance and EMD. 
Note: Needed only if point cloud autoencoder will be trained. Compiling is tricky since it may require changes on the makefile to setup tensorflow and cuda lib/header paths.
$ cd external/tf_ops/emd
$ make
$ cd ../../..

$ cd external/tf_ops/sampling/
$ make
$ cd ../../..

$ cd external/tf_ops/CD
$ make
$ cd ../../..

Alternate CD and EMD lib
$ cd external/structural_losses
$ make
$ cd ../../

Training:
1) To train pc2pix from scratch:
python3 pc2pix.py -t --category="chair" --ptcloud_ae_weights=model_weights/ptcloud/chair-pt-cloud-stacked-ae-chamfer-5-ae-weights-32.h5


