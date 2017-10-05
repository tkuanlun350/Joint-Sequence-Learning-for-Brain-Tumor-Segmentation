import tensorflow as tf
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from nets import model_unet, model_cmc
from dataset.utils import  norm_image_by_patient, eval_single
import glob
import os
import math
from PIL import Image
import SimpleITK as sitk
import dataset.val_test_load_batch as load_batch
from dataset.utils import writeImage, cc, postprocessing, evaluate

tf.app.flags.DEFINE_boolean(
	'post_processing', False, 'post-processing')
tf.app.flags.DEFINE_string(
	'checkpoint_dir', '/data/CVPR_Release/v2/Logs_cmc/', 'path to checkpoint')
tf.app.flags.DEFINE_string(
	'model', 'unet', 'Model to eval')
FLAGS = tf.app.flags.FLAGS
#checkpoint_dir = './tmp/'
checkpoint_dir = '/data/CVPR_Release/v2/Logs2/'
datasetDir = '/data/CVPR_Release/v2/dataset/validation_original_cmc/'

import numpy as np

def padding(im, shape=(240,240)):
    # im 1 240-16, 240-16, 1
    new_im = np.zeros(shape)
    new_im[8: 240-8, 8: 240-8] = im
    return new_im

img_input = tf.placeholder(tf.float32, shape=(1, 3, 240, 240, 4))
la_input = tf.placeholder(tf.int32, shape=(1, 3, 240, 240, 1))

is_training =tf.placeholder(tf.bool)

folder = glob.glob(datasetDir + '*.tfrecord')
assert len(folder) == 15
im_queues = {}
for f in folder:
    imname = f.split("/")[-1].split(".")[0]
    batch = load_batch.get_batch_cmc(f)
    im_queues[imname] = batch

net = model_cmc.Model()

logits, _ = net.net(img_input, is_training)
logits = [tf.argmax(l, axis=3) for l in logits]
saver = tf.train.Saver()

sess = tf.Session()
dir = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
saver.restore(sess, dir)
sess.run(tf.local_variables_initializer())
print("Model restore!")

num_class = 5
hist = np.zeros((num_class, num_class))
out_slices = []
la_slices = []
complete = np.array([0.0, 0.0, 0.0])
core = np.array([0.0, 0.0, 0.0])
enhancing = np.array([0.0, 0.0, 0.0])
for f in folder:
    ind = 0
    out = []
    la = []
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    imname = f.split("/")[-1].split(".")[0]
    batch = im_queues[imname]
    try:
        while not coord.should_stop():
            ind += 1
            image_batch_seq = []
            label_batch_seq = []
            if (ind == 52):
                image_batch_seq.append(prev_image)
                label_batch_seq.append(prev_label)
                for _ in range(2):
                    image_batch, label_batch = sess.run(batch)
                    image_batch_seq.append(image_batch)
                    label_batch_seq.append(label_batch)
            else:
                for _ in range(3):
                    image_batch, label_batch = sess.run(batch)
                    image_batch_seq.append(image_batch)
                    label_batch_seq.append(label_batch)
                # 3,1,240,240,4
            prev_image = image_batch
            prev_label = label_batch
            image_batch = np.array(image_batch_seq).transpose((1,0,2,3,4))
            label_batch =  np.array(label_batch_seq)
            pred = sess.run(logits, feed_dict={
                img_input: image_batch, 
                is_training: False})
                        
            if FLAGS.post_processing:
                for i in range(3):
                    pred[i] = cc(pred[i][0])
            else:
                for i in range(3):
                    pred[i] = pred[i][0]
            #hist += eval_single(pred.astype(np.int64), label_batch, num_class)
            if ind == 52:
                for i in range(1,3):
                    out.append(pred[i])
                    la.append(label_batch[i][0][:,:,0])
            else:
                for i in range(3):
                    out.append(pred[i])
                    la.append(label_batch[i][0][:,:,0])
    except Exception:
        out = np.array(out).astype(np.int64)
        la = np.array(la).astype(np.int64)
        print("fineish one head", out.shape, la.shape)
        if FLAGS.post_processing:
            out = postprocessing(out)
        out_slices.append(out)
        la_slices.append(la)
        _complete, _core, _enhancing = evaluate(out, la)
        complete += _complete
        core += _core
        enhancing += _enhancing
        #outImage = sitk.GetImageFromArray(out)
        #outImage = sitk.Cast(outImage, sitk.sitkUInt8)
        #sitk.WriteImage(outImage, "./results/"+imname+".nii")
        print("finish")

print("SCORES:")
print("COMPLETE: \n", complete/float(15))
print("CORE: \n", core/float(15))
print("ENHANCING: \n", enhancing/float(15))

for ss, las in zip(out_slices, la_slices):
    for s, l in zip(ss, las):
        hist += eval_single(s.astype(np.int64), l, num_class)

acc_total = np.diag(hist).sum() / hist.sum()
print ('accuracy = %f'%np.nanmean(acc_total))
iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
print ('mean IU  = %f'%np.nanmean(iu))
for ii in range(num_class):
    if float(hist.sum(1)[ii]) == 0:
        acc = 0.0
    else:
        acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f "%(ii, acc))
coord.request_stop()
coord.join(threads)
#sess.close()
