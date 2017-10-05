import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.contrib import rnn
from collections import namedtuple
from math import ceil
import convLSTM_upgrade as convLSTM
import copy

CMC_Params = namedtuple('CMC_Parameters', 
										['img_shape',
                                        'sequence_length',
										 'num_classes',
										 ])
default_params = CMC_Params(
    img_shape=(240, 240),
    sequence_length=3,
    num_classes=5,
)

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, initializer, wd):
  var = _variable_on_cpu(
      name,
      shape,
      initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print('Warning -- You have opted to use the orthogonal_initializer function')
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      print('you have initialized one orthogonal matrix.')
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

def BilinearAdditive(net, rate=2, name=None):
    with tf.variable_scope(name):
        b, w, h, c = net.get_shape().as_list() 
        # 10*30*30*64
        net = tf.image.resize_images(net, [h * rate, w * rate]) 
        # 10*60*60*64
        net = tf.split(net, c, 3) 
        # [(10*60*60*1)... *64]
        # split to every four
        net = [net[i:i + 4] for i in range(0, len(net), 4)]
        # [[(10*60*60*1)... *4], [], ... *16]
        net = [tf.add_n(x) for x in net]
        # [(10*60*60*1)... *16]
        net = tf.concat(net, 3)
        # 10*64*64*16
        return net

def DTS(X, r, name):
    def _phase_shift(I, r=2):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (bsize, a, b, r, r))
        print(X.get_shape())
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        print(X.get_shape())
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
        print(X.get_shape())
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  #bsize, a*r, b*r
        return tf.reshape(X, (bsize, a*r, b*r, 1))
    b, w, h, c = X.get_shape().as_list()
    with tf.variable_scope(name):
        with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
            X = slim.conv2d(X, r*r, [1, 1], scope='conv')
            X = _phase_shift(X, r)
    return X

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None, reuse=False):
        return BilinearAdditive(inputT, 2, name)
        #return DTS(inputT, 2, name)
        strides = [1, stride, stride, 1]
        def get_deconv_filter(f_shape):
            """
                reference: https://github.com/MarvinTeichmann/tensorflow-fcn
            """
            width = f_shape[0]
            heigh = f_shape[0]
            f = ceil(width/2.0)
            c = (2 * f - 1 - f % 2) / (2.0 * f)
            bilinear = np.zeros([f_shape[0], f_shape[1]])
            for x in range(width):
                for y in range(heigh):
                    value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                    bilinear[x, y] = value
            weights = np.zeros(f_shape)
            for i in range(f_shape[2]):
                weights[:, :, i, i] = bilinear

            init = tf.constant_initializer(value=weights,
                                            dtype=tf.float32)

            return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)

        with tf.variable_scope(name, reuse=reuse):
            weights = get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape, strides=strides, padding='SAME')
        return deconv

class Model(object):
    def __init__(self):
        self.params = default_params
    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
		"""Network arg_scope.
		"""
		return unet_arg_scope(weight_decay, data_format=data_format)

    def conv_fuse(self, net, is_training=True, channel=20, reuse=False):                     
        with tf.variable_scope('CMC2', reuse=reuse) as scope:
            kernel3D_2 = _variable_with_weight_decay('weights_3d',
                                            shape=[4, 1, 1, channel, channel],
                                            initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                            wd=None)
            net = tf.nn.conv3d(net, kernel3D_2, [1, 1, 1, 1, 1], padding='VALID')
        return net

    def block(self, inputs, start_channel, reuse=False, is_training=True, rate=1, scope=None):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                          normalizer_params={'is_training': is_training},
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            reuse=reuse):
            net = slim.repeat(inputs, 2, slim.conv2d, start_channel, [3, 3], rate=rate, scope=scope)
            return tf.add(net, inputs)

    def encoder(self, inputs, is_training=True, reuse=False):
        start_channel = 40
        end_points = {}
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training},
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            reuse=reuse):
            # input b X 32 X 100 X channel
            net = slim.repeat(inputs, 2, slim.conv2d, 40, [k_size, k_size], scope='conv1')
            end_points['conv1'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool1') # => 16 X 50
            net = slim.repeat(net, 3, slim.conv2d, 40, [k_size, k_size], scope='conv2') # => 16 X 50
            end_points['conv2'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool2') #  => 8 X 25
            net = slim.repeat(net, 3, slim.conv2d, 40, [k_size, k_size], scope='conv3')
            end_points['conv3'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 40, [k_size, k_size], scope='conv4')
            end_points['conv4'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            net = slim.conv2d(net, 40, [k_size, k_size], scope='conv5')
            net = slim.conv2d(net, 40, [k_size, k_size], scope='conv5_1')
            end_points['conv5'] = net

            return [end_points['conv1'], end_points['conv2'], end_points['conv3'], end_points['conv4'], end_points['conv5']]

    def decoder(self, net, f1, f2, f3, f4, is_training=True, reuse=False):
        start_channel = 40
        batch_size, _, _, _ = net.get_shape().as_list()
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training},
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            reuse=reuse):

            _, out_h, out_w, _ = net.get_shape().as_list()
            net = deconv_layer(net, [2, 2, start_channel, start_channel], [batch_size, out_h*2, out_w*2, start_channel], 2, "up1")
            net = tf.multiply(f4, net)
            net = slim.repeat(net, 2, slim.conv2d, start_channel, [3, 3], scope='conv6')
            net = slim.dropout(net, 0.8, is_training=is_training)
            print(net.get_shape()) #30
            _, out_h, out_w, _ = net.get_shape().as_list()
            net = deconv_layer(net, [2, 2, start_channel, start_channel], [batch_size, out_h*2, out_w*2, start_channel], 2, "up2")
            net = tf.multiply(f3, net)
            net = slim.repeat(net, 2, slim.conv2d, start_channel, [3, 3], scope='conv7')
            net = slim.dropout(net, 0.8, is_training=is_training)
            print(net.get_shape()) # 60
            _, out_h, out_w, _ = net.get_shape().as_list()
            net = deconv_layer(net, [2, 2, start_channel, start_channel], [batch_size, out_h*2, out_w*2, start_channel], 2, "up3")
            net = tf.multiply(f2, net)
            net = slim.repeat(net, 2, slim.conv2d, start_channel, [3, 3], scope='conv8')
            net = slim.dropout(net, 0.8, is_training=is_training)
            print(net.get_shape()) # 120
            _, out_h, out_w, _ = net.get_shape().as_list()
            net = deconv_layer(net, [2, 2, start_channel, start_channel], [batch_size, out_h*2, out_w*2, start_channel], 2, "up4")
            net = tf.multiply(f1, net)
            net = slim.repeat(net, 2, slim.conv2d, start_channel, [3, 3], scope='conv9')
            net = slim.dropout(net, 0.8, is_training=is_training)
            print(net.get_shape())

            return net


    def net(self, inputs, is_training=True):
        # inputs b, T, w, h, c
        print(inputs.get_shape())
        start_channel = 40
        batch_size, seq_len, _, _, _ = inputs.get_shape().as_list()
        end_points = {}
        images_seq = tf.transpose(inputs, [1, 0, 2, 3, 4])
        images_seq = tf.unstack(images_seq)

        fuse_feature1 = [[] for _ in range(self.params.sequence_length)]
        fuse_feature2 = [[] for _ in range(self.params.sequence_length)]
        fuse_feature3 = [[] for _ in range(self.params.sequence_length)]
        fuse_feature4 = [[] for _ in range(self.params.sequence_length)]
        mod = [[] for _ in range(self.params.sequence_length)]
        for i in range(self.params.sequence_length):
            image_mod = tf.split(axis=3, num_or_size_splits=4, value=images_seq[i])
            for j in range(4):
                # share weight or not 
                with tf.variable_scope("MME"+str(j)):
                    modality_feature = self.encoder(image_mod[j], is_training=is_training, reuse=(i>0))
                fuse_feature1[i].append(modality_feature[0])
                fuse_feature2[i].append(modality_feature[1])
                fuse_feature3[i].append(modality_feature[2])
                fuse_feature4[i].append(modality_feature[3])
                mod[i].append(modality_feature[4])
        # fusing MRF
        modality_fused1 = []
        modality_fused2 = []
        modality_fused3 = []
        modality_fused4 = []
        modality_fused = []
        for i in range(self.params.sequence_length):
            for j in range(5):
                with tf.variable_scope("MRF"):
                    if j == 0:
                        concat_classifier = tf.transpose(tf.stack(fuse_feature1[i]), [1,0,2,3,4])
                        modality_fused1.append(tf.squeeze(self.conv_fuse(concat_classifier, is_training=is_training, channel=start_channel, reuse=(i>0 or j>0 )), axis=[1]))
                    elif j == 1:
                        concat_classifier = tf.transpose(tf.stack(fuse_feature2[i]), [1,0,2,3,4])
                        modality_fused2.append(tf.squeeze(self.conv_fuse(concat_classifier, is_training=is_training, channel=start_channel, reuse=(i>0 or j>0)), axis=[1]))
                    elif j == 2:
                        concat_classifier = tf.transpose(tf.stack(fuse_feature3[i]), [1,0,2,3,4])
                        modality_fused3.append(tf.squeeze(self.conv_fuse(concat_classifier, is_training=is_training, channel=start_channel, reuse=(i>0 or j>0)), axis=[1]))
                    elif j == 3:
                        concat_classifier = tf.transpose(tf.stack(fuse_feature4[i]), [1,0,2,3,4])
                        modality_fused4.append(tf.squeeze(self.conv_fuse(concat_classifier, is_training=is_training, channel=start_channel, reuse=(i>0 or j>0)), axis=[1]))
                    elif j == 4:
                        concat_classifier = tf.transpose(tf.stack(mod[i]), [1,0,2,3,4])
                        modality_fused.append(tf.squeeze(self.conv_fuse(concat_classifier, is_training=is_training, channel=start_channel, reuse=(i>0 or j>0)), axis=[1]))

        f_shape = modality_fused[0].get_shape().as_list()
        cell = convLSTM.ConvLSTMCell(start_channel, k_size=3, height=f_shape[1], width=f_shape[2], initializer=orthogonal_initializer())
        enc_cell = copy.deepcopy(cell)
        with tf.variable_scope("ConvLSTM"):
            rnn_encoder_output, enc_state = tf.contrib.rnn.static_rnn(enc_cell, modality_fused, dtype=tf.float32)
        with tf.variable_scope("ConvLSTM_decoder"):
            state = enc_state
            output = []
            prev = None
            for i, inp in enumerate(rnn_encoder_output):
                if prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = prev
                else:
                    inp = rnn_encoder_output[-1]
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                rnn_decoder_output, state = cell(inp, state)
                output.append(rnn_decoder_output)
                prev = rnn_decoder_output

        decoder_outputs = []
        for i in range(self.params.sequence_length):
            with tf.variable_scope("Decoder"):
                decode_feature = self.decoder(output[i], modality_fused1[i], modality_fused2[i], modality_fused3[i], modality_fused4[i], is_training=is_training, reuse=(i>0))    
            with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training},
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            reuse=(i>0)):
                #decode_feature = slim.dropout(decode_feature, 0.5, is_training=is_training)
                conv_classifier = slim.conv2d(decode_feature, self.params.num_classes, [1, 1], activation_fn=None, scope='classify') 
                decoder_outputs.append(conv_classifier)

        return decoder_outputs, end_points
   
    def weighted_losses(self, logits, labels):
        """
        loss_weight = np.array([
            0.10455609709637404, 
            1.0, 
            0.67692774919453469, 
            0.0, 
            1.2299177055835784
        ])
        """
        loss_weight = np.array([
            1.0, 
            1.0, 
            1.0, 
            1.0, 
            1.0
        ])
        labels = tf.transpose(labels, [1, 0, 2, 3, 4])
        labels = tf.unstack(labels)
        loss_list = []
        epsilon = tf.constant(value=1e-10)
        for logit, target in zip(logits, labels):
            logit = tf.reshape(logit, (-1, self.params.num_classes))
            logit = logit + epsilon
            labels_for_eval = tf.reshape(target, (-1, 1))
            
            target = tf.reshape(target, [-1])
            target = tf.reshape(tf.one_hot(target, depth=self.params.num_classes), (-1, self.params.num_classes))
            
            softmax = tf.nn.softmax(logit)
            mean_iou = slim.metrics.streaming_mean_iou(tf.reshape(tf.argmax(softmax, 1), (-1,1)), labels_for_eval, self.params.num_classes)
            
            cross_entropy = -tf.reduce_sum(tf.multiply(target * tf.log(softmax + epsilon), loss_weight), axis=[1])
            # cross_entropy = self.generalised_dice_loss(logit, target)
            loss_list.append(cross_entropy)
        loss_list_p = tf.add_n(loss_list)
        # Use reduce mean
        avg_loss = tf.reduce_mean(loss_list_p)
        tf.add_to_collection('losses', avg_loss)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.losses.add_loss(loss)
        return loss, mean_iou




       