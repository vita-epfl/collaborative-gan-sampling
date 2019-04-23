import numpy as np
import pickle 

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
flags.DEFINE_integer("rollout_steps", 1, "Roll Out Steps. [1]")
flags.DEFINE_float("rollout_rate", 1.0, "Roll Out Rate [1.0]")
flags.DEFINE_string("rollout_method", "sgd", "Rollout Method")
flags.DEFINE_string("teacher_name", "rollout", "teacher options: default | scalized | mcts | rollout")

##Hyper Params Tune
flags.DEFINE_integer("D_LR", 1, "Multiplicative Factor of LR of D[1]")
flags.DEFINE_integer("G_LR", 1, "Multiplicative Factor of LR of G[1]")
flags.DEFINE_integer("D_it", 1, "Updates of D per interation [1]")
flags.DEFINE_integer("G_it", 1, "Updates of G per interation [1]")


FLAGS = flags.FLAGS

if FLAGS.teacher_name == 'default':
    with open('scores/{}/epoch_{}_teacher_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}_inception_score.pickle'.format(
               FLAGS.dataset, FLAGS.epoch, FLAGS.teacher_name, FLAGS.D_LR, FLAGS.G_LR, FLAGS.D_it, FLAGS.G_it), 'rb') as handle:
        incp_score = pickle.load(handle)
else:
    with open('scores/{}/epoch_{}_teacher_{}_rollout_method_{}_rollout_steps_{}_rollout_rate_{:06.5f}_inception_score.pickle'.format(
              FLAGS.dataset, FLAGS.epoch, FLAGS.teacher_name, FLAGS.rollout_method, FLAGS.rollout_steps, FLAGS.rollout_rate), 'rb') as handle:
        incp_score = pickle.load(handle)

print(incp_score)