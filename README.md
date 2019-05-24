# pix2pixHD-TF
A minimal tensorflow implementation of pix2pixHD (https://tcwang0509.github.io/pix2pixHD/).
my codes started from original pix2pix implementation based on https://github.com/prashnani/pix2pix-tensorflow


# Setup

This code has been tested to work on the following environment:
- Ubuntu 14.04
- Tensorflow 1.4.0
- Python 3.6 (Numpy, Scikit-image)
- Cuda 8.0, CuDNN 5.1

1. Clone this repository:
```
git clone https://github.com/tiandiao123/Pix2PixHD-TensorFlow
cd Pix2PixHD-TensorFlow
```
2. The data set I am using is shared in public, but you can create your own data sets, and change pipeline, it will be good to go! 

3. Train 
```
python main.py --dataroot /data1/cuiqingli/data/render_v5/unzipV3 --sample_st 1 --sample_ed 80 \
--batch_size 2 --frame_count 3 --resize_w 256 --resize_h 256
```
4. Test
```
python main.py \
  --mode test \ 
  --test_image_path = ./datasets/facades/val \
  --checkpoint_name = <path to checkpoint>
```

