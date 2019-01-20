pc2pix code implementation in Keras(2.2.3) and Tensorflow (1.11)
surface reconstruction requires blender and meshlab (meshlab server)

Environment tested: Ubunutu Linux 16.04LTS, GPUs tested: 1060, 1080Ti and V100

Datasets (ShapeNet point cloud and rendered images) can be downloaded here:
$ wget ...
$ tar jxvf ...


Pre-compiled latent codes (to speed up point cloud encoder prediction)
Note: Needed only if pc2pix will be trained
$ wget ...
$ tar jxvf ...

Model weights of pc2pix and pt cloud autoencoder
$ wget ...
$ tar jxvf ...


Evaluation codes are in evalutation folder. All rendered images will be at evaluation/data
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


Compile CUDA code for Chamfer Distance and EMD. 
Note: Needed only if point cloud autoencoder will be trained. Compiling is tricky since it may require changes on the makefile to setup tensorflow and cuda lib/header paths.
cd external/tf_ops/emd
make
cd ../../..

cd external/tf_ops/sampling/
make
cd ../../..

cd external/tf_ops/CD
make
cd ../../..

Alternate CD and EMD lib
cd external/structural_losses
make
cd ../../
