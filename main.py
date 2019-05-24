import os
import cv2
import scipy.misc
import numpy as np 
import math
import glob
import random

import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import glob
from data_generator import *
from pix2pix_network import pix2pix_network
from random import shuffle
import skimage.io as io
import os
import argparse
from pair_generator import PairGenerator, Inputs
from tf_data import Dataset

curr_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", dest='num_epochs', type=int, default=200, help="specify number of epochs")
parser.add_argument("--lr", dest='lr', type=float, default=0.0002, help="specify learning rate")
parser.add_argument("--dropout", dest='dropout_rate', type=float, default=0.5, help="specify dropout")
parser.add_argument("--dataset", dest='dataset_name', type=str, default='facades', help="specify dataset name")
parser.add_argument("--train_image_path", dest='train_image_path', type=str, help="specify path to training images")
parser.add_argument("--test_image_path", dest='test_image_path', type=str, help="specify path to test images")
parser.add_argument("--crop_size", dest='input_size', type=int, default=256, help="specify crop size of the final jittered input (sec.6.2 of the pix2pix paper https://arxiv.org/pdf/1611.07004.pdf)")
parser.add_argument("--enlarge_size", dest='enlarge_size', type=int, default=286, help="specify enlargement size from which to generate jittered input (sec.6.2 of the pix2pix paper https://arxiv.org/pdf/1611.07004.pdf)")
parser.add_argument("--out_dir", dest='out_dir', type=str, default=curr_path+'/test_outputs/', help="specify path to training images")
parser.add_argument("--checkpoint_name", dest='checkpoint_name', type=str, help="specify the checkpoint")
parser.add_argument("--mode", dest='mode', type=str, help="specify the checkpoint")
parser.add_argument("--checkpoint_name_preamble", dest='checkpoint_name_preamble', default='', type=str, help="specify the initial naming string for checkpoint name")
parser.add_argument('--dataroot', type=str, default='../data', help='path to training data')
parser.add_argument('--sample_st', type=int, default=-1)
parser.add_argument('--sample_ed', type=int, default=-1)
parser.add_argument('--frame_count', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=4, help='gen and disc batch size')



# -------------------------------------------- data and training set --------------------------------------
parser.add_argument('--n_training_samples', type=int, default=80)
# parser.add_argument('--crop_and_resize', action='store_true', help='center crop and resize input')
parser.add_argument('--resize_w', type=int, default=256)
parser.add_argument('--resize_h', type=int, default=256)

# parser.add_argument('--h_crop', action='store_true', help='horizental center crop, if not set, default is original fix crop')

parser.add_argument('--no_crop', action='store_true', help='no crop')
parser.add_argument('--no_resize', action='store_true', help='no resize')

parser.add_argument('--n_frame_total', type=int, default=8, help='num of frames in a loader')
parser.add_argument('--n_frame_G', type=int, default=3)
parser.add_argument('--n_frame_D', type=int, default=3)

parser.add_argument('--pre_padding', action='store_true', help='auto padding (n_frame_total-1) frames ahead')

parser.add_argument('--crop_h_flag', type=int, default=0, help='0 : pos(420, 0) scale(720, 720), 10: args.crop_pos args.crop_scale, 20: json auto')
parser.add_argument('--crop_pos_x', type=int, default=420 )
parser.add_argument('--crop_pos_y', type=int, default=0 )
parser.add_argument('--crop_scale_h', type=int, default=720 )
parser.add_argument('--crop_scale_w', type=int, default=720 )

parser.add_argument('--scale_aug', action='store_true', help='scale random augmentation')
parser.add_argument('--position_aug', action='store_true', help='position random augmentation')
parser.add_argument('--position_aug_range', type=int, default=300, help='max pixels to shift')
parser.add_argument('--scale_aug_range', type=float, default=1.5, help='max times for up/down scale')
parser.add_argument('--scale', type=float, default=1.0, help='a para cache')

# -------------------------------------------- save path and gpu id -----------------------------------
parser.add_argument('--log_path', type=str, default='../log', help='path of logs')
parser.add_argument('--ckpt_path', type=str, default='../checkpoint', help='path of ckpts')
parser.add_argument('--model_name', type=str, default='auto', help='folder name to save weights')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

# -------------------------------------------- global parameters -----------------------------------
parser.add_argument('--img_width', type=int, default=720)
parser.add_argument('--img_height', type=int, default=1280)

# -------------------------------------------- model setting ---------------------------------
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--padding_type', type=str, default='reflect', help='reflect/replicate/zero')

# -------------------------------------------- training configurations ---------------------------------
parser.add_argument('--iter', type=int, default=0, help='number of iterations for training')
parser.add_argument('--pretrain_iter', type=int, default=0, help='number of iterations for pre-train')
parser.add_argument('--epoch', type=int, default=0, help='number of epochs for training')

parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--model_save_interval', type=int, default=10000, help='save weights interval -> step record')
parser.add_argument('--model_save_interval_replace', type=int, default=1000, help='save weights interval -> latest')
parser.add_argument('--display_interval', type=int, default=1000, help='display images interval')
parser.add_argument('--content_loss_type', type=str, default='mse', help='mse//l1')
parser.add_argument('--base_arch', type=str, default='pix_global', help='unet/pix_global')

parser.add_argument('--use_dropout', action='store_true')
parser.add_argument('--drop_rate', type=float, default=0.5)

# DI
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--discriminator_lambda', type=float, default=0.01, help='LAMBDA for adv loss')
parser.add_argument('--adv_loss_type', type=str, default='mse', help='mse/bce')



# -------------------------------------------- pre-load parameters -------------------------------------
parser.add_argument('--generator_weights', type=str, default='', help='path to generator weights to continue training')
parser.add_argument('--continue_iter', type=int, default=0, help='number of iterations for previous training')
parser.add_argument('--continue_epoch', type=int, default=0, help='number of epochs for previous training')

parser.add_argument('--discriminatorI_weights', type=str, default='', help='path to discriminatorI weights to continue training')
parser.add_argument('--continue_gan_iter', type=int, default=0, help='number of iterations for previous training')

# -------------------------------------------- inference configurations ---------------------------------
parser.add_argument('--infer_out_root', type=str, default='../infer_res', help='results root folder of inference')
parser.add_argument('--to_vid', action='store_true', help='convert output images to videos')
parser.add_argument('--old_single_frame_dataloader_debug', action='store_true', help='-1 ~ -0.5 in dataloader and vid_infer')

parser.add_argument('--no_gt', action='store_true', help='do not load gt folder')
parser.add_argument('--remain_frames', action='store_true', help='remain frames after generating videos')

parser.add_argument('--test_vid_ind_st', type=int, default=80)
parser.add_argument('--test_vid_ind_ed', type=int, default=100)

args = parser.parse_args()


def findAllPairs(args):
    images_count = {}
    

    images_name_map = {}
    for i in range(args.sample_st, args.sample_ed+1):
        input_map = {}
        target_map = {}
        folder_id = i
        folder_path = os.path.join(args.dataroot, str(folder_id))


        sub_folder_path = os.path.join(folder_path, 'PNCC')
        images = glob.glob(sub_folder_path+"/*.png")
        count_len = len(images)
        images_count[i] = count_len

    train_list = []
    target_list = []
    for folder_id in range(args.sample_st, args.sample_ed+1):
    	for image_id in range((images_count[folder_id]-args.frame_count)):
    			train_list.append((folder_id, image_id))
    			target_list.append(folder_id+args.frame_count)


    # print(target_list)
    # print(train_list)


    return train_list, target_list 



def read_frame(file_path, crop=False, ih=0, iw=0, resize=False, rh=0, rw=0, norm=True, bias=1, crop_h_flag=0, args=None):
    print(file_path)
    f = cv2.imread(file_path)
    f = np.array(f)
    f = f.astype(np.float32)
    if norm:
        f = f / 127.5 - bias
    if crop:
        if not crop_h_flag:
            down_px = 0 # gimp: 380-1100
            f = f[400-down_px :1120- down_px , 0:720] # lower crop 1130 
            # center crop 654 x 654 from 720 x 720 -- scale 1.1
            scale = 1.0 
            if scale > 1:
                st = int((720 - 720/scale)/2)
                ed = st+int(720/scale)
                f = f[st:ed, st:ed]


            elif scale > 0 and scale < 1:
                side_len = int(720/scale)
                f_cache = np.zeros((side_len, side_len, 3), dtype=float)
                # pos = [0, 0]
                f_cache -= 1 # color black -> value -1
                st = int((720/scale - 720)/2)
                ed = st+720
                f_cache[st:ed, st:ed] = f
                f = f_cache


        elif crop_h_flag in [10, 20, 21, 30]:

            # f = f[args.crop_pos_x:(args.crop_pos_x+args.crop_scale_h), args.crop_pos_y:(args.crop_pos_y+args.crop_scale_w)]
            f_cache = np.zeros((args.crop_scale_h, args.crop_scale_w, 3), dtype=float)
            f_cache -= 1 # color black -> value -1

            x_l = max([args.crop_pos_x, 0])
            x_r = min([args.crop_pos_x+args.crop_scale_h, f.shape[0]]) # h # scipy.misc.imread -> h x w x c
            y_l = max([args.crop_pos_y, 0])
            y_r = min([args.crop_pos_y+args.crop_scale_w, f.shape[1]]) # w

            f_cache[(x_l-args.crop_pos_x):(x_r-args.crop_pos_x), (y_l-args.crop_pos_y):(y_r-args.crop_pos_y)] = f[x_l:x_r, y_l:y_r]
            f = f_cache

        else:
            ih = f.shape[0]
            iw = f.shape[1]
            crop_size = min([ih, iw])
            f = f[int(ih/2 - crop_size/2) : int(ih/2 + crop_size/2), int(iw/2 - crop_size/2): int(iw/2 + crop_size/2)]

    if args.scale_aug:
        scale = args.scale

        f_cache = np.zeros(f.shape, dtype=float)
        f_cache -= 1
        if scale >= 1:
            f_cache[int(f.shape[0]*(1-1/scale)/2):int(f.shape[0]*((1-1/scale)/2))+int(f.shape[0]*1/scale), int(f.shape[1]*(1-1/scale)/2):int(f.shape[1]*((1-1/scale)/2))+int(f.shape[1]*1/scale)] = cv2.resize(f, (int(f.shape[0]/scale), int(f.shape[1]/scale)), interpolation=cv2.INTER_CUBIC)
        else:
            f_cache = cv2.resize(f, (int(f.shape[0]/scale), int(f.shape[1]/scale)), interpolation=cv2.INTER_CUBIC)[int(f.shape[0]*(1/scale-1)/2):int(f.shape[0]*((1/scale-1)/2+1)), int(f.shape[1]*(1/scale-1)/2):int(f.shape[1]*((1/scale-1)/2+1) )]

        f = f_cache

    if resize: 
        f = cv2.resize(f, (rw, rh), interpolation=cv2.INTER_CUBIC)

    return f






def read_image_and_resize(folder_index, image_index, target_image_index):

        image_index = 1
        folder_index = 1
        target_image_index = 1

        # print("image_index: ")
        # print(image_index)
        # print("folder_index: ")
        # print(folder_index)
        # print("target_image_index: ")
        # print(target_image_index)

        IMAGE_WIDTH = args.resize_w 
        IMAGE_HEIGHT = args.resize_h

        out_x = np.zeros( (IMAGE_HEIGHT, IMAGE_WIDTH, 3*3*args.frame_count)) # channel concatenate.

        # image_index = pair_element[0]
        # folder_index = pair_element[1]
        # target_image_index = pair_element[2]

        sample_folder_full = os.path.join(args.dataroot, str(folder_index))
        crop_h_flag = args.crop_h_flag

        if crop_h_flag == 20:
            # read json f1
            i_file_j = sample_folder_full+'/json/j_1.json'

            fp = open(i_file_j, 'r')
            f_content = fp.read()
            fp.close()
            j = json.loads(f_content)

            hyp_para_crop_w = 1.98
            hyp_para_crop_h = 2.53
            hyp_para_crop_x = 0.5 # in width direction
            hyp_para_crop_y = 1.07

            crop_scale_w = int((j['points'][16][0] - j['points'][0][0]) * hyp_para_crop_w)
            crop_scale_h = int((j['points'][8][1] - 0.5 * j['points'][16][1] - 0.5 * j['points'][0][1] ) * hyp_para_crop_h)
            crop_pos_y = int(j['points'][0][0] - hyp_para_crop_x * (j['points'][16][0] - j['points'][0][0]))
            crop_pos_x = int(0.5*(j['points'][16][1] + j['points'][0][1]) - hyp_para_crop_y * (j['points'][8][1] - 0.5* (j['points'][16][1] + j['points'][0][1]) ) )

        if crop_h_flag == 30: # h=w in crop_h_flag 20
            # read json f1
            i_file_j = sample_folder_full+'/json/j_1.json'

            fp = open(i_file_j, 'r')
            f_content = fp.read()
            fp.close()
            j = json.loads(f_content)

            hyp_para_crop_w = 1.98
            hyp_para_crop_h = 2.53
            hyp_para_crop_x = 0.5 # in width direction
            hyp_para_crop_y = 1.07

            crop_scale_w = int((j['points'][16][0] - j['points'][0][0]) * hyp_para_crop_w)
            crop_scale_h = int((j['points'][16][0] - j['points'][0][0]) * hyp_para_crop_w)
            crop_pos_y = int(j['points'][0][0] - hyp_para_crop_x * (j['points'][16][0] - j['points'][0][0]))
            crop_pos_x = int(0.5*(j['points'][16][1] + j['points'][0][1]) - hyp_para_crop_y * (j['points'][8][1] - 0.5* (j['points'][16][1] + j['points'][0][1]) ) )

        if crop_h_flag == 21: # use points 39 44 55
            # read json f1
            i_file_j = sample_folder_full+'/json/j_1.json'

            fp = open(i_file_j, 'r')
            f_content = fp.read()
            fp.close()
            j = json.loads(f_content)

            points=[j['points'][38], j['points'][43], j['points'][54] ]

            hyp_para_crop_w = 7.815
            hyp_para_crop_h = 8.576
            hyp_para_crop_x = 3.364 # in width direction
            hyp_para_crop_y = 3.822

            crop_scale_w = int((points[1][0] - points[0][0]) * hyp_para_crop_w)
            # self.args.crop_scale_h = int((points[2][1] - 0.5 * points[1][1] - 0.5 * points[0][1] ) * hyp_para_crop_h)
            crop_scale_h = crop_scale_w

            crop_pos_y = int(points[0][0] - hyp_para_crop_x * (points[1][0] - points[0][0]))
            crop_pos_x = int(0.5*(points[1][1] + points[0][1]) - hyp_para_crop_y * (points[2][1] - 0.5* (points[1][1] + points[0][1]) ) )


        for i in range(args.frame_count):
            image_extract_id = image_index + i

            PNCC_folder_path = os.path.join(sample_folder_full, 'PNCC')
            extract_image_path = PNCC_folder_path + "/p_" + str(image_extract_id) + ".png"
            out_x[ :, :, i*9:i*9+3] =  read_frame(extract_image_path, crop=not args.no_crop, ih=args.img_height, iw=args.img_width, resize=True, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=1, crop_h_flag=crop_h_flag, args=args)
            


            dtex_folder_path = os.path.join(sample_folder_full, '3dTex')
            extract_image_path = dtex_folder_path + "/t_" + str(image_extract_id) + ".png"
            out_x[ :, :, i*9+3:i*9+6] =  read_frame(extract_image_path, crop=not args.no_crop, ih=args.img_height, iw=args.img_width, resize=True, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=1, crop_h_flag=crop_h_flag, args=args)

            densepose_folder_path = os.path.join(sample_folder_full, 'densepose')
            extract_image_path = densepose_folder_path + "/f_" + str(image_extract_id) + "_IUV.png"
            out_x[ :, :, i*9+6:i*9+9] =  read_frame(extract_image_path, crop=not args.no_crop, ih=args.img_height, iw=args.img_width, resize=True, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=1, crop_h_flag=crop_h_flag, args=args)


        target_img = ""
        if args.mode == 'train':
            face_folder_path = os.path.join(sample_folder_full, 'face')
            extract_image_path = face_folder_path + "/f_" + str((image_extract_id + args.frame_count -1)) + ".png"

            out_y = read_frame(extract_image_path, crop=not args.no_crop, ih=args.img_height, iw=args.img_width, resize=True, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=1, crop_h_flag=crop_h_flag, args=args)

            # out_x = tf.convert_to_tensor(out_x, np.float32)
            # out_y = tf.convert_to_tensor(out_y, np.float32)
            #out_x = out_x.eval()
            #out_y = out_y.eval()
            return (out_x, out_y)
        else:

            # out_x = tf.convert_to_tensor(out_x, np.float32)

            #out_x = out_x.eval()
            return (out_x, )



# def test_data_loader(args):
# 	# generator = PairGenerator(args)
# 	# iter = generator.get_next_pair()

# 	# for i in range(600):
# 	# 	print(next(iter))

# 	ds = Dataset(args)
# 	model_input = ds.next_element

# 	with tf.Session() as sess:
# 		(train_img, target_img) = sess.run([model_input[0], model_input[1]])
# 		print(train_img.shape)
# 		print(target_img.shape)
		
def parse_function(train_image_ids, target_image_id):
    # print("train image ids: ")
    # print(train_image_ids["arg0:0"])
    # print("target_image_id: ")
    # print(target_image_id["arg1:0"])

    folder_id = train_image_ids[0]
    image_id = train_image_ids[1]
    return read_image_and_resize(image_id, folder_id, target_image_id)


def main(args):
    if args.mode == 'train':
        train_list, target_list = findAllPairs(args)
        target_list = np.array(target_list)
        train_list = np.array(train_list)


        dataset = tf.data.Dataset.from_tensor_slices((train_list, target_list))
        dataset = dataset.shuffle(len(target_list))
        dataset = dataset.map(parse_function, num_parallel_calls=4)
        # dataset = dataset.map(train_preprocess, num_parallel_calls=4)
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(5)
        iterator = dataset.make_initializable_iterator()
        ele1, ele2 = iterator.get_next()
        init_op = iterator.initializer


        with tf.Session() as sess:
            sess.run(init_op)
            (train_img, target_img) = sess.run([ele1, ele2])
            #print(new_ele)
            print(train_img.shape)
            print(target_img.shape)
    elif args.mode == 'test':
        pass
    else:
        pass


 #    elif args.mode == 'test':
	# 	test(args)
	# else:
	# 	raise 'mode input should be train or test.'

	

if __name__ == '__main__':
    main(args)