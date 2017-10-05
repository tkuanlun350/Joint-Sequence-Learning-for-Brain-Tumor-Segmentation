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
from dataset.utils import writeImage, writeMedicalImage, fast_hist
import scipy.ndimage

from PIL import Image

tf.app.flags.DEFINE_string(
 'train_data', '/data/CVPR_Release/Brats17TrainingData/',
 'Directory of the datasets')

tf.app.flags.DEFINE_string(
 'val_data', '/data/CVPR_Release/Brats17ValidationData/',
 'Directory of the datasets')

tf.app.flags.DEFINE_string(
 'path_save', '/data/CVPR_Release/v2/dataset/training/',
 'Dictionary to save sythtext record')

tf.app.flags.DEFINE_boolean(
 'is_training', True,
 'save train data or val data')

FLAGS = tf.app.flags.FLAGS

## SythText datasets is too big to store in a record. 
## So Transform tfrecord according to dir name

def _convert_to_example(image_data, label):
	#print 'shape: {}, height:{}, width:{}'.format(shape,shape[0],shape[1])
	example = tf.train.Example(features=tf.train.Features(feature={
			'image/encoded': bytes_feature(image_data),
			'label/encoded': bytes_feature(label)
			}))
	return example
	

def _processing_image(seq, label, depth):
	seqs = []
	labs = []
	for d in [depth-2, depth-1, depth]:
		label_data = label[d]
		labs.append(np.array(label_data))
		mod = []
		for im in seq:
			image_data = im[d]
			#image_data = scipy.ndimage.interpolation.zoom(image_data, 2, order=1, mode='nearest')
			# upsample 
			mod.append(image_data)
		seqs.append(np.array(mod))
	seqs = np.array(seqs)
	labs = np.array(labs)
	return seqs.tobytes(), labs.tobytes()

def _processing_image_single(seq, label, depth):
	mod = []
	for im in seq:
		image_data = im[depth]
		#image_data = scipy.ndimage.interpolation.zoom(image_data, 2, order=1, mode='nearest')
		# upsample 
		mod.append(image_data)
	mod = np.array(mod)
	label_data = label[depth]
	# upsample
	#label_data = scipy.ndimage.interpolation.zoom(label_data, 2, order=1, mode='nearest')
	shape = list(mod.shape)
	#if (mod.shape[0] != 4 or mod.shape[1] != 240 or mod.shape[2] != 240):
	#	print(shape)
	#	print("SHIT")
	#	exit()
	# 4,240,240
	return mod.tobytes(), label_data.tobytes(), shape

def norm_image_by_patient(imname):
	im = sitk.GetArrayFromImage(sitk.ReadImage(imname)).astype(np.float32)
	return (im - im.mean()) / im.std()
	roi_index = im > 0
	mean = im[roi_index].mean()
	std = im[roi_index].std()
	im[roi_index] -= mean
	im[roi_index] /= std
	return im

def count_class_freq(label_batch):
  hist = np.zeros(5)
  imagesPresent = [0,0,0,0,0]
  for i in range(len(label_batch)):
    new_hist = np.bincount(label_batch[i], minlength=5)
    hist += new_hist
    for ii in range(5):
        if (new_hist[ii] != 0):
            imagesPresent[ii] += 1
  print(hist)
  freqs = [hist[v]/float((imagesPresent[v]+1e-5)*240*240) for v in range(5)]
  median = np.median(freqs)
  o = []
  for i in range(5):
      if (freqs[i] <= 1e-5):
          o.append(0.0)
      else:
          o.append(float(median)/(freqs[i]))
  print(o)
  return o

def checkLabel(label, d):
	if np.count_nonzero(label[d]) > 0:
		return True, 1
	else:
		return False, 0

def count_freq(labels):
	freq = np.array([0.0,0.0,0.0,0.0,0.0])
	for la in labels:
		freq += np.bincount(la, minlength=5).astype(np.float32)
	print(freq)
	print(freq/freq.sum())
	count_class_freq(labels)

def run():
	folderHGG = glob.glob(FLAGS.train_data + 'HGG/*')
	folderLGG = glob.glob(FLAGS.train_data + 'LGG/*')	
	folder_train = folderHGG[:-10] + folderLGG[:-5]
	folder_val = folderHGG[-10:] + folderLGG[-5:]
	tf_filename = FLAGS.path_save+'train_cmc_original2.tfrecord'
	all_example = []
	print("Saving training record....")
	all_label_data = []
	with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
		for index, i in enumerate(folder_train):
			print(index)
			imname = i.split("/")[-1]
			flair = glob.glob(i + '/*flair.nii')
			t2 = glob.glob(i + '/*t2.nii')
			t1 = glob.glob(i + '/*t1.nii')
			t1c = glob.glob(i + '/*t1ce.nii')
			t1 = [_t1 for _t1 in t1 if not _t1 in t1c]
			label = glob.glob(i + '/*seg*.nii')[0]
			label = sitk.GetArrayFromImage(sitk.ReadImage(label)).astype(np.float32)
			seq = [norm_image_by_patient(flair[0]),
			norm_image_by_patient(t2[0]),
			norm_image_by_patient(t1[0]),
			norm_image_by_patient(t1c[0])]
			ind = 0
			for depth in range(2,155):
				is_valid, sample_num = checkLabel(label, depth)
				if ( not is_valid):
					continue
				for i in range(sample_num):
					image_data, label_data = _processing_image(seq, label, depth)
					#all_label_data.append(label[depth].flatten().astype(np.int64))
					example = _convert_to_example(image_data, label_data)
					all_example.append(example)
				#tfrecord_writer.write(example.SerializeToString()) 
		#count_freq(all_label_data)
		print("slices:", len(all_example))
		shuffle(all_example)
		for ex in all_example:
			tfrecord_writer.write(ex.SerializeToString()) 
		# [0.011868184281122324, 1.0859737711507338, 0.80660914716121235, 0.0, 1.0]
	
	print 'Transform to tfrecord finished'
	print("Saving validation record....")
	for index, i in enumerate(folder_val):
		imname = i.split("/")[-1]
		print(imname)
		tf_filename = "/data/CVPR_Release/v2/dataset/validation_original_cmc/"+imname+'.tfrecord'
		with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
			imname = i.split("/")[-1]
			flair = glob.glob(i + '/*flair.nii')
			t2 = glob.glob(i + '/*t2.nii')
			t1 = glob.glob(i + '/*t1.nii')
			t1c = glob.glob(i + '/*t1ce.nii')
			t1 = [_t1 for _t1 in t1 if not _t1 in t1c]
			label = glob.glob(i + '/*seg.nii')[0]
			label = sitk.GetArrayFromImage(sitk.ReadImage(label)).astype(np.float32)
			seq = [norm_image_by_patient(flair[0]),
			norm_image_by_patient(t2[0]),
			norm_image_by_patient(t1[0]),
			norm_image_by_patient(t1c[0])]
			ind = 0
			for depth in range(155):
				ind += 1
				image_data, label_data, shape = _processing_image_single(seq, label, depth)
				example = _convert_to_example(image_data, label_data)
				tfrecord_writer.write(example.SerializeToString()) 
	print 'Transform to tfrecord finished'

if __name__ == '__main__':
	run()




