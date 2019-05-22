import os
import glob
import random
import tensorflow as tf
from tensorflow import Tensor


class Inputs(object):
    def __init__(self, train_img, target_img):
        self.train_img = train_img
        self.target_img = target_img


class PairGenerator(object):
    def __init__(self, args):
        self.dataroot = args.dataroot
        self.start_folder_id = args.sample_st
        self.end_folder_id = args.sample_ed
        self.frame_count = args.frame_count
        self.mode = args.mode
        self.images_count = self.get_images_data_stat(self.dataroot)

    def get_images_data_stat(self, dataroot):
        # generates a dictionary between a person and all the photos of that person

        images_count = {}
        

        images_name_map = {}
        for i in range(start_folder_id, end_folder_id+1):
            input_map = {}
            target_map = {}
            folder_id = i
            folder_path = os.path.join(dataroot, folder_id)


            sub_folder_path = os.path.join(folder_path， 'PNCC')
            images = glob.glob(sub_folder_path+"/*.png")
            count_len = len(images)
            images_count[i] = count_len

        return images_count 



    def get_next_pair(self):

        random_folder_index = end_folder_id
        random_image_id = self.images_count[end_folder_id]
        count = 6001
        map = {}
        folder_num = end_folder_id - start_folder_id+1;
        folder_names = ['PNCC', '3dTex', 'densepose']
        array = [i for i in range(start_folder_id, end_folder_id+1)]

        while True:
            if count >= 6000:
                map = {}
                count = 0
                array = [i for i in range(start_folder_id, end_folder_id+1)]
                
                random_folder_index = 0
                random.suffle(array)
                for folder_name in array:
                    image_count = self.images_count[folder_name]
                    temp_array = [i+1 for i in range(images_count-self.frame_count)] 
                    random.suffle(temp_array)
                    map[folder_name] = temp_array

                count = 0
                random_image_id = 0
            count += 1
            folder_path = os.path.join(self.dataroot, str(array[random_folder_index]))
            image_index = map[array[random_folder_index]][random_image_id]

            random_image_id+=1
            if random_folder_index>=len(map[array[random_folder_index]]):
                random_folder_index += 1
                random_image_id  = 0




            yield ({'image_index': image_index, 
                    'folder_index': array[random_folder_index], 
                    'target_image_index': (image_index_index + self.frame_count-1)})







