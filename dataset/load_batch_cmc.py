# *_* coding:utf-8 *_*
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tensorflow as tf 
import numpy as np
slim = tf.contrib.slim

train_data_path = '/data/CVPR_Release/v2/dataset/training'

def get_batch_val(filename,
			  batch_size=5,
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
    with tf.name_scope('cmc/label'):
        label = tf.decode_raw(features['label/encoded'], tf.float32)
        label = tf.transpose(tf.reshape(label, [3,1,240,240]), (0,2,3,1))
    with tf.name_scope('cmc/image'):
        image = tf.decode_raw(features['image/encoded'], tf.float32)
        image = tf.transpose(tf.reshape(image, [3,4,240,240]), (0,2,3,1))
    #image = tf.reshape(image, [32, 100, 3])
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    sh_images, sh_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=1,
            capacity=100 * batch_size,
            min_after_dequeue=batch_size)

    return sh_images, sh_labels

def get_batch(dataset_dir,
			  num_readers=2,
			  batch_size=5,
			  net=None,
			  FLAGS=None,
			  file_pattern = '*.tfrecord',
			  is_training = True,
			  shuffe = False):

    filename = train_data_path+'/train_cmc_original2.tfrecord'
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=50)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                features={
                    'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                    'label/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                })  # return image and label
    with tf.name_scope('cmc/label'):
        label = tf.decode_raw(features['label/encoded'], tf.float32)
        label = tf.transpose(tf.reshape(label, [3,1,240,240]), (0,2,3,1))
    with tf.name_scope('cmc/image'):
        image = tf.decode_raw(features['image/encoded'], tf.float32)
        image = tf.transpose(tf.reshape(image, [3,4,240,240]), (0,2,3,1))
    #image = tf.reshape(image, [32, 100, 3])
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    print("image", image.get_shape(), label.get_shape())
    """
    if (np.random.choice([0, 1]) == 1):
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    """
        
    do_a_crop_flip = tf.random_uniform([], seed=None)
    do_a_crop_flip = tf.greater(do_a_crop_flip, 0.5)
    image = tf.cond(do_a_crop_flip, lambda: tf.reverse_v2(image, [2]),
                                       lambda: image)
    label = tf.cond(do_a_crop_flip, lambda: tf.reverse_v2(label, [2]),
                                       lambda: label)

    sh_images, sh_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=1,
            capacity=100 * batch_size,
            min_after_dequeue=batch_size)

    return sh_images, sh_labels