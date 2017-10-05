## create script that download datasets and transform into tf-record
## Assume the datasets is downloaded into following folders
## mjsyth datasets(41G)
## data/sythtext/*

import numpy as np 
import scipy.io as sio
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tensorflow as tf
import re
from dataset.dataset_utils import int64_feature, float_feature, bytes_feature ,ImageCoder, norm
import glob
import SimpleITK as sitk
from random import shuffle
from dataset.utils import writeImage, writeMedicalImage

from PIL import Image

tf.app.flags.DEFINE_string(
 'train_data', '/data/CVPR_Release/Brats17TrainingData/',
 'Directory of the datasets')

tf.app.flags.DEFINE_string(
 'val_data', '/data/CVPR_Release/Brats17ValidationData/',
 'Directory of the datasets')

tf.app.flags.DEFINE_string(
 'path_save', '/data/CVPR_Release/v2/dataset/testing_original/',
 'Dictionary to save sythtext record')

tf.app.flags.DEFINE_boolean(
 'is_training', True,
 'save train data or val data')

FLAGS = tf.app.flags.FLAGS

## SythText datasets is too big to store in a record. 
## So Transform tfrecord according to dir name

def _convert_to_example(image_data, shape, imname):
	#print 'shape: {}, height:{}, width:{}'.format(shape,shape[0],shape[1])
	example = tf.train.Example(features=tf.train.Features(feature={
			'image/height': int64_feature(shape[1]),
			'image/width': int64_feature(shape[2]),
			'image/channels': int64_feature(shape[0]),
			'image/shape': int64_feature(shape),
			'image/format': bytes_feature('nii'),
			'image/encoded': bytes_feature(image_data),
			'image/name': bytes_feature(imname),
			}))
	return example
	

def _processing_image(seq, depth):
	mod = []
	for im in seq:
		image_data = im[depth]
		mod.append(image_data)
	mod = np.array(mod)
	shape = list(mod.shape)
	if (mod.shape[0] != 4 or mod.shape[1] != 240 or mod.shape[2] != 240):
		print(shape)
		print("SHIT")
		exit()
	# 4,240,240
	return mod.tobytes(), shape

def norm_image_by_patient(imname):
	im = sitk.GetArrayFromImage(sitk.ReadImage(imname)).astype(np.float32)
	return (im - im.mean()) / im.std()
	roi_index = im > 0
	mean = im[roi_index].mean()
	std = im[roi_index].std()
	im[roi_index] -= mean
	im[roi_index] /= std
	print(im[roi_index].mean(), im[roi_index].std())
	return im

def checkLabel(label, d):
	return np.count_nonzero(label[d]) > 100

def run():
    folder = glob.glob(FLAGS.val_data + '*')
    folder = [f for f in folder if '.csv' not in f]
    for index, i in enumerate(folder):
        imname = i.split("/")[-1]
        print(imname)
        tf_filename = FLAGS.path_save+imname+'.tfrecord'
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            flair = glob.glob(i + '/*flair.nii')
            t2 = glob.glob(i + '/*t2.nii')
            t1 = glob.glob(i + '/*t1.nii')
            t1c = glob.glob(i + '/*t1ce.nii')
            t1 = [_t1 for _t1 in t1 if not _t1 in t1c]
            seq = [norm_image_by_patient(flair[0]),
            norm_image_by_patient(t2[0]),
            norm_image_by_patient(t1[0]),
            norm_image_by_patient(t1c[0])]
            for depth in range(155):
                image_data, shape = _processing_image(seq, depth)
                example = _convert_to_example(image_data, shape, imname)
                tfrecord_writer.write(example.SerializeToString()) 
    print 'Transform to tfrecord finished'

if __name__ == '__main__':
	run()




