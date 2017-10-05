"""
Train scripts

"""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tf_utils
from deployment import model_deploy
import dataset.load_batch as load_batch
import dataset.load_batch_patch as load_batch_patch
import dataset.load_batch_cmc as load_batch_cmc

import pickle
from nets import model_unet, model_patch, model_cmc
from tensorflow.contrib.slim.python.slim.learning import train_step
from tensorflow.python.framework import ops
# from beholder.beholder import Beholder
slim = tf.contrib.slim

tf.app.flags.DEFINE_string('model', 'unet',
							'Model to use') # unet, patch, CMC
# =========================================================================== #
# Text Network flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
	'loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float(
	'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
	'match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_string(
	'file_pattern', '*.tfrecord', 'tf_record pattern')

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
	'train_dir', '/data/CVPR_Release/v2/Logs2',
	'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('num_clones', 1,
							'Number of model clones to deploy.')
tf.app.flags.DEFINE_integer('shuffle_data', False,
							'Wheather shuffe the datasets')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
							'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
	'num_ps_tasks', 0,
	'The number of parameter servers. If the value is 0, then the parameters '
	'are handled locally by the worker.')
tf.app.flags.DEFINE_integer(
	'num_readers', 2,
	'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
	'num_preprocessing_threads', 4,
	'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
	'log_every_n_steps', 100,
	'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
	'save_summaries_secs', 60,
	'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
	'save_interval_secs', 60*10,
	'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
	'gpu_memory_fraction', 0.90
	, 'GPU memory fraction to use.')
tf.app.flags.DEFINE_integer(
	'task', 0, 'Task id of the replica running the training.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
	'weight_decay', 0.0005, 'The weight decay on the model weights_1.')
tf.app.flags.DEFINE_string(
	'optimizer', 'adam',
	'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
	'"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
	'adadelta_rho', 0.95,
	'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
	'adagrad_initial_accumulator_value', 0.1,
	'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
	'adam_beta1', 0.9,
	'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
	'adam_beta2', 0.999,
	'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
						  'The learning rate power.')
tf.app.flags.DEFINE_float(
	'ftrl_initial_accumulator_value', 0.1,
	'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
	'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
	'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
	'momentum', 0.9,
	'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
	'learning_rate_decay_type',
	'fixed',
	'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
	' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
	'end_learning_rate', 0.001,
	'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
	'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
	'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
	'num_epochs_per_decay', 1,
	'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
	'moving_average_decay', None,
	'The decay to use for the moving average.'
	'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_boolean(
	'use_batch', False,
	'Wheather use batch_norm or not')
tf.app.flags.DEFINE_boolean(
	'use_hard_neg', True,
	'Wheather use use_hard_neg or not')
tf.app.flags.DEFINE_boolean(
	'use_whiten', True,
	'Wheather use whiten or nbot,genally you can choose whiten or batchnorm tech.')
tf.app.flags.DEFINE_float('clip_gradient_norm', 0,
                   'If greater than 0 then the gradients would be clipped by '
                   'it.')
# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
	'dataset_name', 'sythtext', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
	'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
	'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
	'dataset_dir', "/data/CVPR_Release/v2/dataset/shuffle/", 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
	'labels_offset', 0,
	'An offset for the labels in the dataset. This flag is primarily used to '
	'evaluate the VGG and ResNet architectures which do not use a background '
	'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
	'model_name', 'text_box_300', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
	'data_format', 'NHWC', 'data format.')
tf.app.flags.DEFINE_string(
	'preprocessing_name', None, 'The name of the preprocessing to use. If left '
	'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
	'batch_size', 10, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
	'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', 40000,
							'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('num_samples', 40000,
							'Num of training set')
# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
	'checkpoint_path', None,
	'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
	'checkpoint_model_scope', None,
	'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
	'checkpoint_exclude_scopes', None,
	'Comma-separated list of scopes of variables to exclude when restoring '
	'from a checkpoint.')
tf.app.flags.DEFINE_string(
	'trainable_scopes', None,
	'Comma-separated list of scopes to filter the set of variables to train.'
	'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
	'ignore_missing_vars', False,
	'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean(
    'fine_tune', False,
    'Weather use fine_tune')
tf.app.flags.DEFINE_integer(
    'validation_check', 1000,
    'frequency to eval'
)

FLAGS = tf.app.flags.FLAGS

# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()
        if (FLAGS.model == 'unet'):
            print("Use UNET Model")
            net = model_unet.Model()
        elif (FLAGS.model == 'patch'):
            print("Use Patch Model")
            net = model_patch.Model()
        elif (FLAGS.model == 'cmc'):
            print("Use CMC Model")
            net = model_cmc.Model()

        with tf.device(deploy_config.inputs_device()):
            if (FLAGS.model == 'unet'):
                batch_queue = \
                load_batch.get_batch(FLAGS.dataset_dir,
                                        FLAGS.num_readers,
                                        FLAGS.batch_size,
                                        None,
                                        FLAGS,
                                        file_pattern = FLAGS.file_pattern,
                                        is_training = True,
                                        shuffe = FLAGS.shuffle_data)
            elif (FLAGS.model == 'patch'):
                batch_queue = \
                load_batch_patch.get_batch(FLAGS.dataset_dir,
                                        FLAGS.num_readers,
                                        FLAGS.batch_size,
                                        None,
                                        FLAGS,
                                        file_pattern = FLAGS.file_pattern,
                                        is_training = True,
                                        shuffe = FLAGS.shuffle_data)
            elif (FLAGS.model == 'cmc'):
                batch_queue = \
                load_batch_cmc.get_batch(FLAGS.dataset_dir,
                                        FLAGS.num_readers,
                                        FLAGS.batch_size,
                                        None,
                                        FLAGS,
                                        file_pattern = FLAGS.file_pattern,
                                        is_training = True,
                                        shuffe = FLAGS.shuffle_data)

        # =================================================================== #
        # Define the model running on every GPU.
        # =================================================================== #
        def clone_fn(batch_queue):
            batch_shape = [1]*3
            b_image, label = batch_queue
            
            logits, end_points = net.net(b_image)

            # Add loss function.
            loss, mean_iou = net.weighted_losses(logits, label)
            return end_points, mean_iou


        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)


        end_points, mean_iou = clones[0].outputs
        update_ops.append(mean_iou[1])
        #for end_point in end_points:
        #	x = end_points[end_point]
        #	summaries.add(tf.summary.histogram('activations/' + end_point, x))


        for loss in tf.get_collection('EXTRA_LOSSES',first_clone_scope):
            summaries.add(tf.summary.scalar(loss.op.name, loss))

        #
        #for variable in slim.get_model_variables():
        #	summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = tf_utils.configure_learning_rate(FLAGS,
                                                                FLAGS.num_samples,
                                                                global_step)
            optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if FLAGS.fine_tune:
            gradient_multipliers = pickle.load(open('nets/multiplier_300.pkl','rb'))
        else:
            gradient_multipliers = None
            

        if FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = tf_utils.get_variables_to_train(FLAGS)

        #  and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))
        if gradient_multipliers:
            with ops.name_scope('multiply_grads'):
                clones_gradients = slim.learning.multiply_gradients(clones_gradients, gradient_multipliers)

        if FLAGS.clip_gradient_norm > 0:
            with ops.name_scope('clip_grads'):
                clones_gradients = slim.learning.clip_gradient_norms(clones_gradients, FLAGS.clip_gradient_norm)
        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                    global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                            name='train_op')

        #train_tensor = slim.learning.create_train_op(total_loss, optimizer, gradient_multipliers=gradient_multipliers)
        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                            first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # =================================================================== #
        # Kicks off the training.
        # =================================================================== #
        
        def train_step_fn(session, *args, **kwargs):
            # visualizer = Beholder(session=session, logdir=FLAGS.train_dir)
            total_loss, should_stop = train_step(session, *args, **kwargs)

            if train_step_fn.step % FLAGS.validation_check == 0:
                _mean_iou = session.run(train_step_fn.mean_iou)
                print('evaluation step %d - loss = %.4f mean_iou = %.2f%%' %\
                 (train_step_fn.step, total_loss, _mean_iou ))
            # evaluated_tensors = session.run([end_points['conv4'], end_points['up1']])
            # example_frame = session.run(end_points['up2'])
            # visualizer.update(arrays=evaluated_tensors, frame=example_frame)

            train_step_fn.step += 1
            return [total_loss, should_stop]

        train_step_fn.step = 0
        train_step_fn.end_points = end_points
        train_step_fn.mean_iou = mean_iou[0]


        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction,
                                    allocator_type="BFC")
        config = tf.ConfigProto(gpu_options=gpu_options,
                                log_device_placement=False,
                                allow_soft_placement = True,								
                                inter_op_parallelism_threads = 0,
                                intra_op_parallelism_threads = 1,)
        saver = tf.train.Saver(max_to_keep=5,
                                keep_checkpoint_every_n_hours=1.0,
                                write_version=2,
                                pad_step_number=False)


        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            master='',
            is_chief=True,
            train_step_fn=train_step_fn,
            init_fn=tf_utils.get_init_fn(FLAGS),
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            saver=saver,
            save_interval_secs=FLAGS.save_interval_secs,
            session_config=config,
            sync_optimizer=None)

if __name__ == '__main__':
	tf.app.run()


