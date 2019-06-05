# pix2pixHD-TF
A minimal tensorflow implementation of pix2pixHD (https://tcwang0509.github.io/pix2pixHD/).
my codes started from original pix2pix implementation based on https://github.com/prashnani/pix2pix-tensorflow


# Setup

This code has been tested to work on the following environment:
- Ubuntu 14.04
- Tensorflow 0.11
- Python 2.7 (Numpy, Scikit-image)
- Cuda 8.0, CuDNN 5.1

1. Clone this repository:
```
git clone https://github.com/prashnani/pix2pix-TF.git
cd pix2pix-TF
```
2. Download a dataset using the download script provided with the [original code from authors](https://github.com/phillipi/pix2pix/blob/master/datasets/download_dataset.sh):
```
cd datasets
bash ./download_dataset.sh facades
```
NOTE: This script downloads the images provided by the authors of pix2pix which are processed versions of the original dataset. Specifically, for example in the [facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/), the images in both domains (domain B: image of the buildings and domain A: semantic labels for the facades) are resized to 256x256 pixels and appended along width so that the final image is of size 512x256. This input is used for the image translation task: domain A -> domain B.

3. Train 
```
python main2.py --dataroot /data1/cuiqingli/data/render_v5/denseposeV4_256 --sample_st 1 --sample_ed 80 --num_epochs 10 \
--batch_size 2 --frame_count 11 --resize_w 256 --resize_h 256 --mode train --lr 0.0002 --model_name model256_denseposeV4_mse --gpu_ids 0 --need_resize False 
```
NOTE: For the generator, the U-net encoder-decoder is implemented. For the discriminator, a 70x70 discriminator is implemented. Both these choices have been experimentally proven to be the best compared to some of the alternatives by the authors of pix2pix. Also, as specified in the errata in the [pix2pix paper](https://arxiv.org/pdf/1611.07004.pdf) (Sec.6.3), batchnorm is removed from bottleneck layer. 

4. Test
```
python main2.py --dataroot /data1/cuiqingli/data/render_v5/denseposeV4_256 --sample_st 81 --sample_ed 100 \
--batch_size 2 --frame_count 11 --resize_w 256 --resize_h 256 --mode test --checkpoint_name /data1/cuiqingli/Pix2PixHD-TensorFlow/checkpoints/model256_denseposeV4 --gpu_ids 0 \
--epoch_id_inference 9 --model_name model256_dneseposeV4
```

