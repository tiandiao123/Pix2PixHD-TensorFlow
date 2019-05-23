import tensorflow as tf
from tensorflow import Tensor
# from pair_generator import PairGenerator, Inputs
import os
import cv2
import scipy.misc
import numpy as np 
import math
import glob
import random

import json


class Dataset(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.frame_count = args.frame_count
        self.args = args
        self.dataroot = args.dataroot
        self.bias = 1
        self.mode = args.mode


        # self.dataroot = args.dataroot
        self.start_folder_id = args.sample_st
        self.end_folder_id = args.sample_ed
        self.frame_count = args.frame_count
        self.mode = args.mode
        self.images_count = self.get_images_data_stat(self.dataroot)

        # mygenerator = self.generator
        
        self.next_element = self.build_iterator(self.generator)

    def build_iterator(self, pair_gen):
        
        prefetch_batch_buffer = 5

        dataset = tf.data.Dataset.from_generator(pair_gen,
                                                 output_types=(tf.int64, tf.int64, tf.int64))
        dataset = dataset.map(self._read_image_and_resize)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(prefetch_batch_buffer)
        iter = dataset.make_one_shot_iterator()
        res = iter.get_next()

        return res


    def generator(self):
        random_folder_index = self.end_folder_id
        random_image_id = self.images_count[self.end_folder_id]
        count = 6001
        map = {}
        folder_num = self.end_folder_id - self.start_folder_id+1;
        folder_names = ['PNCC', '3dTex', 'densepose']
        array = [i for i in range(self.start_folder_id, self.end_folder_id+1)]

        while True:
            if count >= 6000:
                map = {}
                count = 0
                array = [i for i in range(self.start_folder_id, self.end_folder_id+1)]
                
                random_folder_index = 0
                random.shuffle(array)
                for folder_name in array:
                    image_count = self.images_count[folder_name]
                    temp_array = [i+1 for i in range(image_count-self.frame_count)] 
                    random.shuffle(temp_array)
                    map[folder_name] = temp_array

                count = 0
                random_image_id = 0
            count += 1
            folder_path = os.path.join(self.dataroot, str(array[random_folder_index]))
            image_index = map[array[random_folder_index]][random_image_id]

            random_image_id+=1
            if random_image_id>=len(map[array[random_folder_index]]):
                random_folder_index += 1
                random_image_id  = 0

            print((image_index, array[random_folder_index], (image_index + self.frame_count-1)))
            yield image_index, array[random_folder_index], (image_index + self.frame_count-1)




    def get_images_data_stat(self, dataroot):
        images_count = {}
        images_name_map = {}
        for i in range(self.start_folder_id, self.end_folder_id+1):
            input_map = {}
            target_map = {}
            folder_id = i
            folder_path = os.path.join(dataroot, str(folder_id))


            sub_folder_path = os.path.join(folder_path, 'PNCC')
            images = glob.glob(sub_folder_path+"/*.png")
            count_len = len(images)
            images_count[i] = count_len

        return images_count

    def _read_image_and_resize(self, *arguments):
        
        print("arguments: ----------------------------------------------------")
        for i in range(len(arguments)):
            print(arguments[i])

        image_index = 1
        folder_index = 1
        target_image_index = 1

        # print("image_index: ")
        # print(image_index)
        # print("folder_index: ")
        # print(folder_index.numpy())
        # print("target_image_index: ")
        # print(target_image_index.numpy())

        IMAGE_WIDTH = self.args.resize_w 
        IMAGE_HEIGHT = self.args.resize_h

        out_x = np.zeros( (IMAGE_HEIGHT, IMAGE_WIDTH, 3*3*self.frame_count)) # channel concatenate.

        # image_index = pair_element[0]
        # folder_index = pair_element[1]
        # target_image_index = pair_element[2]

        sample_folder_full = os.path.join(self.dataroot, str(folder_index))
        crop_h_flag = self.args.crop_h_flag

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

            self.args.crop_scale_w = int((j['points'][16][0] - j['points'][0][0]) * hyp_para_crop_w)
            self.args.crop_scale_h = int((j['points'][8][1] - 0.5 * j['points'][16][1] - 0.5 * j['points'][0][1] ) * hyp_para_crop_h)
            self.args.crop_pos_y = int(j['points'][0][0] - hyp_para_crop_x * (j['points'][16][0] - j['points'][0][0]))
            self.args.crop_pos_x = int(0.5*(j['points'][16][1] + j['points'][0][1]) - hyp_para_crop_y * (j['points'][8][1] - 0.5* (j['points'][16][1] + j['points'][0][1]) ) )

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

            self.args.crop_scale_w = int((j['points'][16][0] - j['points'][0][0]) * hyp_para_crop_w)
            self.args.crop_scale_h = int((j['points'][16][0] - j['points'][0][0]) * hyp_para_crop_w)
            self.args.crop_pos_y = int(j['points'][0][0] - hyp_para_crop_x * (j['points'][16][0] - j['points'][0][0]))
            self.args.crop_pos_x = int(0.5*(j['points'][16][1] + j['points'][0][1]) - hyp_para_crop_y * (j['points'][8][1] - 0.5* (j['points'][16][1] + j['points'][0][1]) ) )

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

            self.args.crop_scale_w = int((points[1][0] - points[0][0]) * hyp_para_crop_w)
            # self.args.crop_scale_h = int((points[2][1] - 0.5 * points[1][1] - 0.5 * points[0][1] ) * hyp_para_crop_h)
            self.args.crop_scale_h = self.args.crop_scale_w

            self.args.crop_pos_y = int(points[0][0] - hyp_para_crop_x * (points[1][0] - points[0][0]))
            self.args.crop_pos_x = int(0.5*(points[1][1] + points[0][1]) - hyp_para_crop_y * (points[2][1] - 0.5* (points[1][1] + points[0][1]) ) )


        for i in range(self.frame_count):
            image_extract_id = image_index + i

            PNCC_folder_path = os.path.join(sample_folder_full, 'PNCC')
            extract_image_path = PNCC_folder_path + "/p_" + str(image_extract_id) + ".png"
            out_x[ :, :, i*9:i*9+3] =  read_frame(extract_image_path, crop=not self.args.no_crop, ih=self.args.img_height, iw=self.args.img_width, resize=True, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=self.bias, crop_h_flag=crop_h_flag, args=self.args)
            


            dtex_folder_path = os.path.join(sample_folder_full, '3dTex')
            extract_image_path = dtex_folder_path + "/t_" + str(image_extract_id) + ".png"
            out_x[ :, :, i*9+3:i*9+6] =  read_frame(extract_image_path, crop=not self.args.no_crop, ih=self.args.img_height, iw=self.args.img_width, resize=True, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=self.bias, crop_h_flag=crop_h_flag, args=self.args)

            densepose_folder_path = os.path.join(sample_folder_full, 'densepose')
            extract_image_path = densepose_folder_path + "/f_" + str(image_extract_id) + "_IUV.png"
            out_x[ :, :, i*9+6:i*9+9] =  read_frame(extract_image_path, crop=not self.args.no_crop, ih=self.args.img_height, iw=self.args.img_width, resize=True, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=self.bias, crop_h_flag=crop_h_flag, args=self.args)


        target_img = ""
        if self.mode == 'train':
            face_folder_path = os.path.join(sample_folder_full, 'face')
            extract_image_path = face_folder_path + "/f_" + str((image_extract_id + self.frame_count -1)) + ".png"

            out_y = read_frame(extract_image_path, crop=not self.args.no_crop, ih=self.args.img_height, iw=self.args.img_width, resize=True, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=self.bias, crop_h_flag=crop_h_flag, args=self.args)

            out_x = tf.convert_to_tensor(out_x, np.float32)
            out_y = tf.convert_to_tensor(out_y, np.float32)
            #out_x = out_x.eval()
            #out_y = out_y.eval()
            return (out_x, out_y)
        else:

            out_x = tf.convert_to_tensor(out_x, np.float32)

            #out_x = out_x.eval()
            return (out_x, )

        # random_image_id+=1
        # if random_folder_index>=len(map[array[random_folder_index]]):
        #     random_folder_index += 1
        #     random_image_id  = 0


        # # read images from disk
        # img1_file = tf.read_file(pair_element[PairGenerator.person1])
        # img2_file = tf.read_file(pair_element[PairGenerator.person2])

        # img1 = tf.image.decode_image(img1_file)
        # img2 = tf.image.decode_image(img2_file)

        # # let tensorflow know that the loaded images have unknown dimensions, and 3 color channels (rgb)
        # img1.set_shape([None, None, 3])
        # img2.set_shape([None, None, 3])

        # # resize to model input size
        # img1_resized = tf.image.resize_images(img1, target_size)
        # img2_resized = tf.image.resize_images(img2, target_size)

        # pair_element[self.img1_resized] = img1_resized
        # pair_element[self.img2_resized] = img2_resized
        # pair_element[self.label] = tf.cast(pair_element[PairGenerator.label], tf.float32)

        # return pair_element




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