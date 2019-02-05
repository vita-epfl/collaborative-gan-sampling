import os
import scipy.misc
import numpy as np

from dcgan_cs import DCGAN
from dcgan_utils import pp, visualize, to_json, show_all_variables, make_folders

import tensorflow as tf
flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 28, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("save_model", True, "True for saving model, False for nothing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
flags.DEFINE_string("teacher_name", "default", "teacher options: default | scalized | mcts | rollout | gpurollout")

flags.DEFINE_float("D_LR", 1.0, "Multiplicative Factor of LR of D[1]")
flags.DEFINE_integer("G_LR", 1, "Multiplicative Factor of LR of G[1]")
flags.DEFINE_integer("D_it", 1, "Updates of D per interation [1]")
flags.DEFINE_integer("G_it", 1, "Updates of G per interation [1]")
flags.DEFINE_integer("num_it", 1, "Number of runs [1]")

##Set the following parameters
flags.DEFINE_string("mode", "training", "mode options: training | refinement | testing")
flags.DEFINE_boolean("denoise", False, "True for denoising, False for not denoising [False]")
flags.DEFINE_boolean("use_refined", True, "True for shaping using refined samples, False for using default samples [False]")
flags.DEFINE_integer("epoch", 5, "Epoch to train G and D [20]/ Epochs to refine D [5]")
flags.DEFINE_integer("load_epoch", 0, "Epoch to load from for refinement")
flags.DEFINE_string("load_model_dir", "dc_checkpoints/celebA/epoch_5_teacher_default_rollout_method_momentum_rollout_steps_100_rollout_rate_50.00000/celebA_64_64_64/", "directory to load model from")
flags.DEFINE_integer("refine_D_iters", 1, "Number of iteration to refine D [4]")
flags.DEFINE_boolean("save_figs", True, "True for saving the comparison figures, False for nothing [False]")
flags.DEFINE_integer("rollout_steps", 100, "Roll Out Steps. [100]")
flags.DEFINE_integer("rollout_rate", 50, "Roll Out Rate [50]")
flags.DEFINE_string("rollout_method", "momentum", "Rollout Method: sgd | momentum")
'''
mode: If training, trains the GAN. If refinement, shapes the  discriminator. If testing, collaboratively samples  
denoise: Sets the application to Image Denoising
use_refined: Uses the refined samples to shape the discriminator. If False, uses the default generated samples  
epoch: Total number of epochs to run the `mode`
load_epoch: If not training, the iterations/epoch to load the saved model from
load_model_dir: The directory of saved model
refine_D_iters: Used in refinement mode, to switch from 'refinement' to 'testing' upon complete of `refine_D_iters` of shaping D
rollout_steps: The number of rollout steps (k)
rollout_rate: The step_size of each rollout step  
rollout_method: The optimization algorithm to roll out the samples
'''


FLAGS = flags.FLAGS

# checkpoints/mnist/epoch_21_teacher_default_rollout_method_sgd_rollout_steps_100_rollout_rate_10.00000/mnist_64_28_28/
# checkpoints/mnist/epoch_5_teacher_default_rollout_method_sgd_rollout_steps_100_rollout_rate_10.00000/mnist_64_28_28/
# dc_checkpoints/mnist/epoch_12_teacher_gpurollout_rollout_method_sgd_rollout_steps_1_rollout_rate_1.00000/mnist_64_28_28/
# dc_checkpoints/celebA/epoch_17_teacher_gpurollout_rollout_method_sgd_rollout_steps_1_rollout_rate_1.00000/celebA_64_64_64/

def main(_):
  # pp.pprint(flags.FLAGS.__flags)

  if FLAGS.dataset == 'celebA':
    FLAGS.input_height = 108
    FLAGS.crop = True
    FLAGS.output_height = 64

  if FLAGS.mode == 'training':
    FLAGS.teacher_name = 'default'
  else:
    FLAGS.teacher_name = 'gpurollout'


  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height
  
  make_folders(FLAGS, dcgan=True)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        z_dim=FLAGS.generate_test_images,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        data_dir=FLAGS.data_dir,
        mode=FLAGS.mode,
        config=FLAGS)

    # show_all_variables()

    if FLAGS.train:
      net_inc_score = []
      net_fid_score = []
      for i in range(FLAGS.num_it):
        inc_score, fid_score = dcgan.train(FLAGS)
        net_inc_score.append(inc_score)
        net_fid_score.append(fid_score)
      print("Average Summary")
      ave_inc_score = np.mean(np.array(net_inc_score), axis=0)
      ave_fid_score = np.mean(np.array(net_fid_score), axis=0)
      np.set_printoptions(precision=2)
      print("Avg Inception Score")
      print(ave_inc_score)
      print("Avg FID Score")
      print(ave_fid_score)
      dcgan.save_scores(ave_inc_score, ave_fid_score, config=FLAGS, num_it=FLAGS.num_it)

    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      

    # Below is code for visualization
    OPTION = 0
    visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
