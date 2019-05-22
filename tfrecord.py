import numpy as np
import skimage.io as io
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


tfrecords_filename = './datasets/data.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)


filename_pairs = []

filename_pairs = [[1, 2], [2, 3]]
for ele1, ele2 in filename_pairs:
	print(ele1 + " " +ele2)
	


