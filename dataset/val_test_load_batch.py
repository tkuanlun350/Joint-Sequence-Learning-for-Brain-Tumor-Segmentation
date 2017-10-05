# *_* coding:utf-8 *_*
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tensorflow as tf 
slim = tf.contrib.slim

train_data_path = '/data/CVPR_Release/v2/dataset/shuffle'

tf.app.flags.DEFINE_string(
 'train_data', '/data/CVPR_Release/v2/dataset/shuffle',
 'Directory of the datasets')

def get_batch_cmc(filename,
			  batch_size=1,
			  shuffe = False):

    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                features={
                    'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                    'label/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                })  # return image and label
    with tf.name_scope('unet/label'):
        label = tf.decode_raw(features['label/encoded'], tf.float32)
        label = tf.transpose(tf.reshape(label, [1,240,240]), (1,2,0))
    with tf.name_scope('unet/image'):
        image = tf.decode_raw(features['image/encoded'], tf.float32)
        image = tf.transpose(tf.reshape(image, [4,240,240]), (1,2,0))
    #image = tf.reshape(image, [32, 100, 3])
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    sh_images, sh_labels = tf.train.batch(
            [image, label], batch_size=batch_size, num_threads=1,
            capacity=100 * batch_size)

    return sh_images, sh_labels

def get_batch_val(filename,
			  batch_size=1,
			  shuffe = False):

    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                features={
                    'image/height': tf.FixedLenFeature([1], tf.int64),
                    'image/width': tf.FixedLenFeature([1], tf.int64),
                    'image/channels': tf.FixedLenFeature([1], tf.int64),
                    'image/shape': tf.FixedLenFeature([3], tf.int64),
                    'image/format': tf.FixedLenFeature([], tf.string, default_value='jpeg'),
                    'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                    'image/name': tf.VarLenFeature(dtype = tf.string),
                    'label/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                })  # return image and label
    shape = features['image/shape']
    with tf.name_scope('unet/label'):
        label = tf.decode_raw(features['label/encoded'], tf.float32)
        label = tf.transpose(tf.reshape(label, [1,240,240]), (1,2,0))
    with tf.name_scope('unet/image'):
        image = tf.decode_raw(features['image/encoded'], tf.float32)
        image = tf.transpose(tf.reshape(image, [4,240,240]), (1,2,0))
    #image = tf.reshape(image, [32, 100, 3])
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    sh_images, sh_labels = tf.train.batch(
            [image, label], batch_size=batch_size, num_threads=1,
            capacity=100 * batch_size)

    return sh_images, sh_labels

def get_batch_test(filename,
			  batch_size=1,
			  shuffe = False):

    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                features={
                    'image/height': tf.FixedLenFeature([1], tf.int64),
                    'image/width': tf.FixedLenFeature([1], tf.int64),
                    'image/channels': tf.FixedLenFeature([1], tf.int64),
                    'image/shape': tf.FixedLenFeature([3], tf.int64),
                    'image/format': tf.FixedLenFeature([], tf.string, default_value='jpeg'),
                    'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                    'image/name': tf.VarLenFeature(dtype = tf.string),
                })  # return image and label
    with tf.name_scope('unet/image'):
        image = tf.decode_raw(features['image/encoded'], tf.float32)
        image = tf.transpose(tf.reshape(image, [4,240,240]), (1,2,0))
    image = tf.cast(image, tf.float32)
    sh_images = tf.train.batch(
            [image], batch_size=batch_size, num_threads=1,
            capacity=155 * batch_size)

    return sh_images