import numpy as np
import tensorflow as tf
import skimage.io as io
import skimage.transform as transform
from random import randint
from random import shuffle
import os
import cv2
import scipy.misc
import numpy as np 
import math
import json

#images loaded in "paired" setting
def load_images_paired(img_names,is_train=True, true_size = 256, enlarge_size = 286):
  if is_train:
    resize_to = enlarge_size
  else:
    resize_to = true_size
  
  A_imgs = np.zeros((len(img_names),true_size,true_size,3)) # ASSUMING RGB FOR NOW
  B_imgs = np.zeros((len(img_names),true_size,true_size,3)) # ASSUMING RGB FOR NOW
  iter = 0
  for name in img_names:
    paired_im = io.imread(name)
    # print name
    B = transform.resize(paired_im[:,0:true_size,:],[resize_to,resize_to,3])*2.0-1.0
    A = transform.resize(paired_im[:,true_size:true_size*2,:],[resize_to,resize_to,3])*2.0-1.0
    tl_h = randint(0,resize_to-true_size)
    tl_w = randint(0,resize_to-true_size)
    flipflag = randint(0,1)>0 and is_train
    A_imgs[iter,:,:,:] = flip_image(A[tl_h:tl_h+true_size,tl_w:tl_w+true_size,:],flipflag)
    B_imgs[iter,:,:,:] = flip_image(B[tl_h:tl_h+true_size,tl_w:tl_w+true_size,:],flipflag)
    # io.imsave('A.png',(A+1)/2)
    iter += 1  
  
  return A_imgs,B_imgs

def read_frame(file_path, crop=False, ih=0, iw=0, resize=False, rh=0, rw=0, norm=True, bias=1, crop_h_flag=0, args=None):
  # print(file_path)
  f = cv2.imread(file_path)
  f = np.array(f)
  f = f.astype(np.float32)
  if norm:
      f = f / 127.5 - bias
  # print("resize value : " + str(resize))
  # print("resize_image_flag : " + str(args.need_resize))
  if resize == False:
    return f
  
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

def read_image_and_resize(args, folder_index, image_index, target_image_index):


        IMAGE_WIDTH = args.resize_w 
        IMAGE_HEIGHT = args.resize_h

        out_x = np.zeros( (IMAGE_HEIGHT, IMAGE_WIDTH, 3*3*args.frame_count)) # channel concatenate.
        out_y = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))

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

            args.crop_scale_w = int((j['points'][16][0] - j['points'][0][0]) * hyp_para_crop_w)
            args.crop_scale_h = int((j['points'][8][1] - 0.5 * j['points'][16][1] - 0.5 * j['points'][0][1] ) * hyp_para_crop_h)
            args.crop_pos_y = int(j['points'][0][0] - hyp_para_crop_x * (j['points'][16][0] - j['points'][0][0]))
            args.crop_pos_x = int(0.5*(j['points'][16][1] + j['points'][0][1]) - hyp_para_crop_y * (j['points'][8][1] - 0.5* (j['points'][16][1] + j['points'][0][1]) ) )

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

            args.crop_scale_w = int((j['points'][16][0] - j['points'][0][0]) * hyp_para_crop_w)
            args.crop_scale_h = int((j['points'][16][0] - j['points'][0][0]) * hyp_para_crop_w)
            args.crop_pos_y = int(j['points'][0][0] - hyp_para_crop_x * (j['points'][16][0] - j['points'][0][0]))
            args.crop_pos_x = int(0.5*(j['points'][16][1] + j['points'][0][1]) - hyp_para_crop_y * (j['points'][8][1] - 0.5* (j['points'][16][1] + j['points'][0][1]) ) )

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

            args.crop_scale_w = int((points[1][0] - points[0][0]) * hyp_para_crop_w)
            # self.args.crop_scale_h = int((points[2][1] - 0.5 * points[1][1] - 0.5 * points[0][1] ) * hyp_para_crop_h)
            args.crop_scale_h = crop_scale_w

            args.crop_pos_y = int(points[0][0] - hyp_para_crop_x * (points[1][0] - points[0][0]))
            args.crop_pos_x = int(0.5*(points[1][1] + points[0][1]) - hyp_para_crop_y * (points[2][1] - 0.5* (points[1][1] + points[0][1]) ) )


        for i in range(args.frame_count):
            image_extract_id = image_index + i

            PNCC_folder_path = os.path.join(sample_folder_full, 'PNCC')
            extract_image_path = PNCC_folder_path + "/p_" + str(image_extract_id) + ".png"
            out_x[ :, :, i*9:i*9+3] =  read_frame(extract_image_path, crop=not args.no_crop, ih=args.img_height, iw=args.img_width, resize=args.need_resize, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=1, crop_h_flag=crop_h_flag, args=args)
            


            dtex_folder_path = os.path.join(sample_folder_full, '3dTex')
            extract_image_path = dtex_folder_path + "/t_" + str(image_extract_id) + ".png"
            out_x[ :, :, i*9+3:i*9+6] =  read_frame(extract_image_path, crop=not args.no_crop, ih=args.img_height, iw=args.img_width, resize=args.need_resize, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=1, crop_h_flag=crop_h_flag, args=args)

            densepose_folder_path = os.path.join(sample_folder_full, 'densepose')
            extract_image_path = densepose_folder_path + "/f_" + str(image_extract_id) + "_IUV.png"
            out_x[ :, :, i*9+6:i*9+9] =  read_frame(extract_image_path, crop=not args.no_crop, ih=args.img_height, iw=args.img_width, resize=args.need_resize, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=1, crop_h_flag=crop_h_flag, args=args)


        target_img = ""
        if args.mode == 'train':
            face_folder_path = os.path.join(sample_folder_full, 'face')
            extract_image_path = face_folder_path + "/f_" + str(target_image_index) + ".png"

            out_y = read_frame(extract_image_path, crop=not args.no_crop, ih=args.img_height, iw=args.img_width, resize=args.need_resize, \
                rh=IMAGE_HEIGHT, rw=IMAGE_WIDTH, bias=1, crop_h_flag=crop_h_flag, args=args)

            # out_x = tf.convert_to_tensor(out_x, np.float32)
            # out_y = tf.convert_to_tensor(out_y, np.float32)
            #out_x = out_x.eval()
            #out_y = out_y.eval()
            return (out_x, out_y)
        else:

            # out_x = tf.convert_to_tensor(out_x, np.float32)

            #out_x = out_x.eval()
            return (out_x, out_y)

def load_images_paired2(args, cur_index, batch_size, image_size, frame_count, train_list):
  A_imgs = np.zeros((batch_size,image_size,image_size,9*args.frame_count)) # ASSUMING RGB FOR NOW
  B_imgs= np.zeros((batch_size, image_size, image_size, 3))


  for batch_id in range(batch_size):
    print(cur_index+batch_id)
    image_index = train_list[cur_index+batch_id][1]
    folder_index = train_list[cur_index+batch_id][0]
    target_image_index = train_list[cur_index+batch_id][2]

    out_x, out_y = read_image_and_resize(args, folder_index, image_index, target_image_index)
    A_imgs[batch_id,:,:,:] = out_x
    B_imgs[batch_id,:,:,:] = out_y


  return A_imgs, B_imgs







def flip_image(img,flipflag):
  if flipflag:
    return np.fliplr(img)
  else:
    return img

#----------------- TF record Data IO ------------------
# using code from: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

def preprocess_images_tf(imgs):
    assert len(imgs.get_shape().as_list()) == 3
    # imgs_max = tf.reduce_max(imgs,axis=[0,1,2],keep_dims=True)

    imgs_normed = tf.subtract(tf.multiply(tf.divide(tf.to_float(imgs,name='ToFloat'),tf.constant(255.0)),tf.constant(2.0)),tf.constant(1.0))
    return imgs_normed

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tfrecord_data_paired(out_file_name, hazy_txt_file, clear_txt_file):

    writer = tf.python_io.TFRecordWriter(out_file_name)

    hazy_im_names = utils.get_text_file_lines(hazy_txt_file)
    clear_im_names = utils.get_text_file_lines(clear_txt_file)

    image_name_pairs = zip(hazy_im_names,clear_im_names)
    print(image_name_pairs)
    shuffle(image_name_pairs)

    for hazy_name, clear_name in image_name_pairs:
      print(clear_name)
      hazy_img = np.array(io.imread(hazy_name[:-1])) # [:-1] because \n needs to be removed from the name # convert RGB to BGR im1 = im1[:,:,::-1]
      hazy_img = np.array(np.multiply(transform.resize(hazy_img, [286,286]),255)).astype('uint8')
      clear_img = np.array(io.imread(clear_name[:-1]))
      clear_img = np.array(np.multiply(transform.resize(clear_img, [286,286]),255)).astype('uint8')
      print(type(clear_img[0][0][0]))
      print(clear_img.shape)
      h,w,c = hazy_img.shape

      hazy_str = hazy_img.tostring()
      clear_str = clear_img.tostring()
      print(len(clear_str))
      # write the record
      record = tf.train.Example(features=tf.train.Features(feature={
      'h': _int64_feature(h),
      'w': _int64_feature(w),
      'hazy_str_raw': _bytes_feature(hazy_str),
      'clear_str_raw': _bytes_feature(clear_str)}))

      writer.write(record.SerializeToString())

    writer.close()

def decode_tfrecord_data_paired(file_name, batch_size_val, out_h, out_w):
    out_channel = 3

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name)
    features = tf.parse_single_example(
      serialized_example,      
      features={
        'h': tf.FixedLenFeature([], tf.int64),
        'w': tf.FixedLenFeature([], tf.int64),
        'hazy_str_raw': tf.FixedLenFeature([], tf.string),
        'clear_str_raw': tf.FixedLenFeature([], tf.string)
        })    

    img1_vec = tf.decode_raw(features['hazy_str_raw'], tf.uint8)
    img2_vec = tf.decode_raw(features['clear_str_raw'], tf.uint8)
    
    h = tf.cast(features['h'], tf.int32)
    w = tf.cast(features['w'], tf.int32)

    image_shape = tf.stack([h, w, out_channel])    

    img1 = tf.reshape(img1_vec, image_shape)
    print(img1.get_shape().as_list())
    img1_normed = preprocess_images_tf(img1)

    img2 = tf.reshape(img2_vec, image_shape)    
    img2_normed = preprocess_images_tf(img2)
            
    off_h = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2
    off_w = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2
    tformed_img1 = tf.image.crop_to_bounding_box(image=img1_normed,
                                           offset_height = off_h, 
                                           offset_width = off_w, 
                                           target_height=out_h,
                                           target_width=out_w)
    
    tformed_img2 = tf.image.crop_to_bounding_box(image=img2_normed,
                                           offset_height = off_h,
                                           offset_width = off_w,
                                           target_height=out_h,
                                           target_width=out_w)
    
    
    images1, images2 = tf.train.shuffle_batch( [tformed_img1, tformed_img2],
                                                 batch_size=batch_size_val,
                                                 capacity=50, # can be changed to more appropriate values
                                                 num_threads=2,
                                                 min_after_dequeue=30)    
    return images1, images2

def write_tfrecord_data(out_file_name, image_names, image_dir): # NOTE! to be used only for unpaired GAN training

    writer = tf.python_io.TFRecordWriter(out_file_name)
    shuffle(image_names)
    for image_name in image_names:
        print(image_dir+image_name[0:-1])
        img = np.array(io.imread(image_dir+image_name[0:-1])) # convert RGB to BGR im1 = im1[:,:,::-1]        
        img = transform.resize(img, [286,286])
        
        h,w,c = img.shape

        img_str = img.tostring()
        # write the record
        record = tf.train.Example(features=tf.train.Features(feature={
        'h': _int64_feature(h),
        'w': _int64_feature(w),
        'img_str_raw': _bytes_feature(img_str),
        'img_name': _bytes_feature(image_name)}))

        writer.write(record.SerializeToString())

    writer.close()

def get_features(file_name):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name)
    features = tf.parse_single_example(
      serialized_example,      
      features={
        'h': tf.FixedLenFeature([], tf.int64),
        'w': tf.FixedLenFeature([], tf.int64),
        'img_str_raw': tf.FixedLenFeature([], tf.string),
        'img_name': tf.FixedLenFeature([], tf.string)
        })    
    return features


def decode_tfrecord_data_two_inputs(file_name1, file_name2, batch_size_val, out_h, out_w, paired=True):
    out_channel = 3

    features1 = get_features(file_name1)
    features2 = get_features(file_name2)

    img1_vec = tf.decode_raw(features1['img_str_raw'], tf.uint8)
    img2_vec = tf.decode_raw(features2['img_str_raw'], tf.uint8)
    
    img1_name = tf.cast(features1['img_name'], tf.string)
    img2_name = tf.cast(features2['img_name'], tf.string)

    h = tf.cast(features1['h'], tf.int32)
    w = tf.cast(features1['w'], tf.int32)

    image_shape = tf.stack([h, w, out_channel])    

    img1 = tf.reshape(img1_vec, image_shape)
    img1_normed = preprocess_images_tf(img1)

    img2 = tf.reshape(img2_vec, image_shape)    
    img2_normed = preprocess_images_tf(img2)
            
    off_h = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2
    off_w = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2
    tformed_img1 = tf.image.crop_to_bounding_box(image=img1_normed,
                                           offset_height = off_h, 
                                           offset_width = off_w, 
                                           target_height=out_h,
                                           target_width=out_w)
    
    tformed_img2 = tf.image.crop_to_bounding_box(image=img2_normed,
                                           offset_height = off_h,
                                           offset_width = off_w,
                                           target_height=out_h,
                                           target_width=out_w)
    
    
    images1, images2 = tf.train.shuffle_batch( [tformed_img1, tformed_img2],
                                                 batch_size=batch_size_val,
                                                 capacity=50, # can be changed to more appropriate values
                                                 num_threads=2,
                                                 min_after_dequeue=50)    
    return images1, images2

def decode_tfrecord_data_single_input(input_file_name, batch_size_val, out_h, out_w):
    out_channel = 3

    # assuming two file names
    file_name = input_file_name

    reader = tf.TFRecordReader()

    features1 = get_features(file_name1)

    img1_vec = tf.decode_raw(features1['img_str_raw'], tf.uint8)
    
    img1_name = tf.cast(features1['img_name'], tf.string)

    h = tf.cast(features1['h'], tf.int32)
    w = tf.cast(features1['w'], tf.int32)
   
    image_shape = tf.pack([h, w, out_channel])    
    
    img1 = tf.reshape(img1_vec, image_shape)
    img1_normed = preprocess_images_tf(img1)

    off_h = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2
    off_w = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2

    tformed_img1 = tf.image.crop_to_bounding_box(image=img1_normed,
                                           offset_height = off_h, #this can be changed to add "jitter" as per pix2pix paper
                                           offset_width = off_w, #this can be changed to add "jitter" as per pix2pix paper
                                           target_height=out_h,
                                           target_width=out_w)
        
    
    images1 = tf.train.shuffle_batch(tformed_img1,
                                                 batch_size=batch_size_val,
                                                 capacity=30, # can be changed to more appropriate values
                                                 num_threads=2,
                                                 min_after_dequeue=10)    
    return images1