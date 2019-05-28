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
from pix2pixHD_network import pix2pixHD_network
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
parser.add_argument("--dropout_rate", dest='dropout_rate', type=float, default=0.5, help="specify dropout")
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
parser.add_argument('--cur_path', type=str, default="", help="current path")



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
parser.add_argument('--epoch', type=int, default=10, help='number of epochs for training')

parser.add_argument('--model_save_interval', type=int, default=10000, help='save weights interval -> step record')
parser.add_argument('--model_save_interval_replace', type=int, default=1000, help='save weights interval -> latest')
parser.add_argument('--display_interval', type=int, default=1000, help='display images interval')
parser.add_argument('--content_loss_type', type=str, default='mse', help='mse//l1')
parser.add_argument('--base_arch', type=str, default='pix_global', help='unet/pix_global')

parser.add_argument('--use_dropout', action='store_true')
parser.add_argument('--drop_rate', type=float, default=0.5)




args = parser.parse_args()


def findAllTuple(args):
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
    # target_list = []
    for folder_id in range(args.sample_st, args.sample_ed+1):
    	for image_index in range((images_count[folder_id]-args.frame_count)):
            image_id = image_index + 1
            train_list.append([folder_id, image_id, image_id+args.frame_count-1])
    			

    return train_list



def test(args):
    ######## data IO
    out_dir = args.out_dir  
    test_image_names = glob.glob(args.test_image_path+'/*.jpg') 
    if not os.path.isdir(out_dir): os.mkdir(out_dir)
    # TF placeholder for graph input
    image_A = tf.placeholder(tf.float32, [None, args.input_size, args.input_size, 3])
    image_B = tf.placeholder(tf.float32, [None, args.input_size, args.input_size, 3])
    keep_prob = tf.placeholder(tf.float32)
    # Initialize model
    model = pix2pixHD_network(image_A,image_B,args.batch_size,keep_prob, args, weights_path='')
    # Loss
    D_loss, G_loss, G_loss_L1, G_loss_GAN = model.compute_loss()
    # Initialize a saver
    saver = tf.train.Saver(max_to_keep=None)
    # Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    ######### Start training
    with tf.Session(config=config) as sess: 
        with tf.device('/gpu:0'):
            # Initialize all variables and start queue runners  
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            # To continue training from one of the checkpoints
            if not args.checkpoint_name:
                raise IOError('In test mode, a checkpoint is expected.')
            saver.restore(sess, args.checkpoint_name)
            # Test network
            print('generating network output')
            for curr_test_image_name in test_image_names:           
                splits = curr_test_image_name.split('/')
                splits = splits[len(splits)-1].split('.')
                print(curr_test_image_name)
                batch_A,batch_B = load_images_paired(list([curr_test_image_name]),
                    is_train = False, true_size = args.input_size, enlarge_size = args.enlarge_size)
                fake_B = sess.run(model.generator_output(image_A), 
                    feed_dict={image_A: batch_A.astype('float32'), image_B: batch_B.astype('float32'), keep_prob: 1-args.dropout_rate})             
                io.imsave(out_dir+splits[0]+'_test_output_fakeB.png',(fake_B[0]+1.0)/2.0)
                io.imsave(out_dir+splits[0]+'_realB.png',(batch_B[0]+1.0)/2.0)
                io.imsave(out_dir+splits[0]+'_realA.png',(batch_A[0]+1.0)/2.0)

def train(args):
    ######## data IO
    train_list = findAllTuple(args)
    shuffle(train_list)

    ######## Training variables
    num_epochs = args.num_epochs
    lr = args.lr    
    batch_size = args.batch_size
    total_train_images = len(train_list)
    num_iters_per_epoch = total_train_images/args.batch_size
    input_h = args.resize_h
    input_w = args.resize_w
    image_size = args.resize_w
    frame_count = args.frame_count
    cur_path = os.getcwd()

    ######### Prep for training
    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = cur_path + "/TBoard_files"
    checkpoint_path = cur_path + "/checkpoints/"
    out_dir = checkpoint_path+'sample_outputs/'
    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
    if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
    if not os.path.isdir(out_dir): os.mkdir(out_dir)

    # TF placeholder for graph input
    image_A = tf.placeholder(tf.float32, [None, input_h, input_w, 9*args.frame_count])
    image_B = tf.placeholder(tf.float32, [None, input_h, input_w, 3])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    model = pix2pixHD_network(image_A,image_B,batch_size, keep_prob, args, weights_path='')

    # Loss
    D_loss, G_loss, G_loss_L1, G_loss_GAN = model.compute_loss()

    # Summary
    tf.summary.scalar("D_loss", D_loss)
    tf.summary.scalar("G_loss_GAN", G_loss_GAN)
    tf.summary.scalar("G_loss_L1", G_loss_L1)
    merged = tf.summary.merge_all()



    # Optimization
    D_vars = [v for v in tf.trainable_variables() if v.name.startswith("dis_")]
    G_vars = [v for v in tf.trainable_variables() if v.name.startswith("gen_")]

    learning_rate = args.lr
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_train_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(D_loss, var_list=D_vars)
        G_train_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss, var_list=G_vars)

    # Initialize a saver and summary writer
    saver = tf.train.Saver(max_to_keep=None)

    # Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    ######### Start training
    with tf.Session(config=config) as sess: 
        with tf.device('/gpu:0'):

            # Initialize all variables and start queue runners  
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            threads = tf.train.start_queue_runners(sess=sess)
            train_writer = tf.summary.FileWriter(filewriter_path + '/train', sess.graph)

            # To continue training from one of the checkpoints
            if args.checkpoint_name:
                saver.restore(sess, args.checkpoint_name)
            
            start_time = time.time()
            # Loop over number of epochs
            start_epoch = 0
            for epoch in range(start_epoch,num_epochs):

                print("generator pretrain epoch {} begin: ".format(str(epoch)))
                step = 0
                generator_loss = 0
                for iter_id in np.arange(0,len(train_list),batch_size):
                    batch_A,batch_B = load_images_paired2(args, iter_id, batch_size, image_size, frame_count, train_list)

                    _, g_loss = sess.run([G_train_opt, G_loss], \
                        feed_dict={image_A: batch_A.astype('float32'), image_B: batch_B.astype('float32'), keep_prob: 0.5})
                    step += 1
                    generator_loss += g_loss

                    average_g_loss = (float)(generator_loss)/step

                    print("the iteration {} of epcoh {}'s for pretrain G_loss is: {}".format(str(step), str(epoch), str(average_g_loss)))


                
                step = 0
                # Loop over iterations of an epoch
                D_loss_accum = 0.0
                G_loss_accum = 0.0

                # Test network
                print('disciminator network comes in and training')
                print("GAN training epcoh {} begins: ".format(str(epoch)))

                for iter_id in np.arange(0,len(train_list),batch_size):

                    # Get a batch of images (paired)                
                    step+=1
                    batch_A,batch_B = load_images_paired2(args, iter_id, batch_size, image_size, frame_count, train_list)

                    _, d_loss = sess.run([D_train_opt, D_loss], feed_dict={image_A: batch_A.astype('float32'), image_B: batch_B.astype('float32'), keep_prob: 0.5})
                    _, g_loss = sess.run([G_train_opt, G_loss], \
                        feed_dict={image_A: batch_A.astype('float32'), image_B: batch_B.astype('float32'), keep_prob: 0.5}) 

                    # Record losses for display
                    D_loss_accum += d_loss
                    G_loss_accum += g_loss
                    average_d_loss = (float)(D_loss_accum)/step
                    average_g_loss = (float)(G_loss_accum)/step
                    print("iteration {} of epcoh {} for GAN training's D_loss {}, G_loss {}".format(str(step),str(epoch),str(average_d_loss), str(average_g_loss)))

                    train_writer.add_summary(summary, epoch*len(train_list)/batch_size + iter_id)
                    step += 1           
                
                end_time = time.time()

                # Save the most recent model
                for f in glob.glob(checkpoint_path+"model_epoch"+str(epoch-1)+"*"):
                    os.remove(f)
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpt')
                save_path = saver.save(sess, checkpoint_name)       

            train_writer.close()




def main(args):
    if args.mode == 'train':
        train(args)
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