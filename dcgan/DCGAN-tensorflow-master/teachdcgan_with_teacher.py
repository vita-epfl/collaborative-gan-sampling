from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import pickle
import _pickle

from ops import *
from dcgan_utils import *
import util_mnist
import inc_score
import sys 

sys.path.append('.')
sys.path.append(os.path.join('..','..', 'teacher'))

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64, eval_size=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir='./data', mode=None, config=None):
    """
    Args:
        Describe Args
    """

    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num
    self.eval_size = eval_size

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.data_dir = data_dir
    self.mode = config.mode


    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')


    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]
    elif self.dataset_name == 'cifar10':
      self.data_X, self.data_y = self.load_cifar10()
      self.c_dim = self.data_X[0].shape[-1]
    else:
      data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
      self.data = glob(data_path)
      if len(self.data) == 0:
        raise Exception("[!] No data found in '" + data_path + "'")
      np.random.shuffle(self.data)
      imreadImg = imread(self.data[0])
      if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1

      if len(self.data) < self.batch_size:
        raise Exception("[!] Entire dataset size is less than the configured batch_size")
    
    self.grayscale = (self.c_dim == 1)
    self.config = config
    self.use_refined = config.use_refined

    self.build_model(config)

  def build_model(self, config):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None


    self.disc_LR = tf.placeholder(tf.float32, (), name='disc_LR')

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')

    #Denoising Setup
    if config.denoise:
      print("Denoising Experiment")
      G                       = self.generator(self.z)
      self.G                  = self.noisy_real(inputs,G)
    
    else:
      self.G                  = self.generator(self.z)

    self.D, self.D_logits   = self.discriminator(inputs, reuse=False)
    self.sampler            = self.sampler(self.z)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    # self.evaluator          = self.evaluator(self.z)
    self.evaluator = tf.placeholder(tf.float32, [None, 28, 28, 1], name='ev')
    self.eval_real = tf.placeholder(tf.float32, [None, 28, 28, 1], name='real_ev')

    if self.dataset_name == 'mnist':
      MNIST_CLASSIFIER_FROZEN_GRAPH = './model/classify_mnist_graph_def.pb'
      self.eval_score = util_mnist.mnist_score(self.evaluator, MNIST_CLASSIFIER_FROZEN_GRAPH)
      self.fid_eval_score =  util_mnist.mnist_frechet_distance(self.evaluator, self.eval_real, MNIST_CLASSIFIER_FROZEN_GRAPH)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
         
    self.d_loss = self.d_loss_real + self.d_loss_fake

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    # build teacher
    print("[!] teacher_name: ", config.teacher_name)

    if config.teacher_name == 'default':
        self.teacher = None 
    elif config.teacher_name == 'scalized':
        from teacher_scalized import TeacherScalized
        self.teacher = TeacherScalized(config)
        print("Scalized Teacher")
    elif config.teacher_name == 'direct':
        from teacher_direct import TeacherDirect
        self.teacher = TeacherDirect(scale=1.0)
        print("Direct Teacher")
    elif config.teacher_name == 'rollout':
        from teacher_rollout import TeacherRollout
        self.teacher = TeacherRollout(config)
        self.teacher.set_env(self, self.sess)
        print("Rollout Teacher") 
    elif config.teacher_name == 'gpurollout':
        from teacher_gpu_rollout import TeacherGPURollout
        self.teacher = TeacherGPURollout(config)
        self.teacher.set_env(self, self.sess)
        print("GPU Rollout Teacher")
    else:
        raise NotImplementedError

    self.saver = tf.train.Saver()
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      ###### D Optimizer ######
      # Create D optimizer.
      self.d_optim = tf.train.AdamOptimizer(self.disc_LR*config.D_LR, beta1=config.beta1)
      # Compute the gradients for a list of variables.
      self.grads_d_and_vars = self.d_optim.compute_gradients(self.d_loss, var_list=self.d_vars)
      self.grad_default_real = self.d_optim.compute_gradients(self.d_loss_real, var_list=inputs)
      # Ask the optimizer to apply the capped gradients.
      self.update_d = self.d_optim.apply_gradients(self.grads_d_and_vars) 
      ## Get Saliency Map - Teacher 
      self.saliency_map = tf.gradients(self.d_loss, self.inputs)[0]

      ###### G Optimizer ######
      # Create G optimizer.
      self.g_optim = tf.train.AdamOptimizer(config.learning_rate*config.G_LR, beta1=config.beta1)
      
      # Compute the gradients for a list of variables.
      ## With respect to Generator Weights - AutoLoss
      self.grad_default = self.g_optim.compute_gradients(self.g_loss, var_list=[self.G, self.g_vars]) 
      ## With Respect to Images given to D - Teacher
      # self.grad_default = g_optim.compute_gradients(self.g_loss, var_list=)
      if config.teacher_name == 'default':
        self.optimal_grad = self.grad_default[0][0]
        self.optimal_batch = self.G - self.optimal_grad 
      else:
        self.optimal_grad, self.optimal_batch = self.teacher.build_teacher(self.G, self.D_, self.grad_default[0][0], self.inputs)

      # Ask the optimizer to apply the manipulated gradients.
      ## AutoLoss
      # update_g = g_optim.apply_gradients(grads_g_and_vars) 
      ## Teacher
      # self.grad_plugin = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, self.c_dim])
      # grads_collected = tf.gradients(self.G, self.g_vars, self.grad_plugin)
      grads_collected = tf.gradients(self.G, self.g_vars, self.optimal_grad)
      grads_and_vars_collected = list(zip(grads_collected, self.g_vars))
      
      self.g_teach = self.g_optim.apply_gradients(grads_and_vars_collected)

  def train(self, config):

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    if config.mode == "training":
      self.plot_every = 1000
    else:
      self.plot_every = 200

    if config.mode == "refinement":
      ckpt_dir = config.load_model_dir
      self.restore(ckpt_dir, config.load_epoch)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]

    elif config.dataset == 'cifar10':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]

    else:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 0
    start_time = time.time()

    self.incp_score = []
    self.fid_score = []
    self.incp_score_refined = []
    self.fid_score_refined = []

    for epoch in xrange(config.epoch):

      ##Switch to testing once D refined
      if epoch == config.refine_D_iters and config.mode != 'training':
        print("Changing to Testing")
        config.mode = 'testing'
      
      ##Load Batchsize
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      elif config.dataset == 'cifar10':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size 
      else:      
        self.data = glob(os.path.join(
          config.data_dir, config.dataset, self.input_fname_pattern))
        np.random.shuffle(self.data)
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      ##Batch GD
      for idx in xrange(0, int(batch_idxs)):
        counter += 1
        ##Load Data
        if config.dataset == 'mnist':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        elif config.dataset == 'cifar10':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        ##Update D according to mode
        for _ in xrange(0, config.D_it):
          self.update_Discriminator(batch_images, batch_z, counter)
        
        ##Update G according to mode
        for _ in xrange(0, config.G_it):
          if config.mode == "training":
            _, g_loss = self.sess.run([self.g_teach, self.g_loss], feed_dict={self.z: batch_z, self.inputs: batch_images, self.disc_LR: self.config.learning_rate})
          elif config.mode == "refinement":
            g_loss = self.sess.run(self.g_loss, feed_dict={self.z: batch_z, self.inputs: batch_images, self.disc_LR: self.config.learning_rate})
          elif config.mode == "testing":
            pass
          else:
            raise NotImplementedError

        # Eval Loss
        errD_fake = self.d_loss_fake.eval({ self.z: batch_z, self.inputs: batch_images, self.disc_LR: self.config.learning_rate })
        errD_real = self.d_loss_real.eval({ self.inputs: batch_images, self.disc_LR: self.config.learning_rate })
        errG = self.g_loss.eval({self.z: batch_z, self.inputs: batch_images, self.disc_LR: self.config.learning_rate })
        # Print loss 
        if (idx % 200) == 0:
          print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, config.epoch, idx, batch_idxs,
              time.time() - start_time, errD_fake+errD_real, errG))

        if counter % self.plot_every == 0:    
          ## If config.mode training, save sample images 
          if config.mode == 'training':
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.disc_LR: self.config.learning_rate,
              },
            )
            if config.save_figs:
              save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            plot_fig_samples(samples)
          
          ##Get Quantitive Scores and Visualize Refined Samples 
          if config.mode == "training":
            self.score_training(counter, epoch, idx)
          else:
            self.score_refinement(batch_z, batch_images, counter, epoch, idx)

      if config.save_model:
        if config.mode == "training" and (epoch % 4 == 0):
          print("Saving Model")
          self.save(config.checkpoint_dir, epoch)

    ## Results
    print("Final Summary: ")
    print("Inception Scores")
    print(self.incp_score)
    if config.dataset == 'mnist':
      print("FID Score ")
      print(self.fid_score)

    print("Mean IS Default: ", np.mean(self.incp_score))
    print("Mean IS Refined: ", np.mean(self.incp_score_refined))
    print("Mean FID Default: ", np.mean(self.fid_score))
    print("Mean FID Refined: ", np.mean(self.fid_score_refined))

    return self.incp_score, self.fid_score



  def update_Discriminator(self, batch_images, batch_z, counter):
    # Update D network
    if self.mode == "training":
      _ = self.sess.run([self.update_d], feed_dict={self.inputs: batch_images, self.z: batch_z, self.disc_LR: self.config.learning_rate})
   
    #Refine D network
    elif self.mode == "refinement":
      if self.use_refined:
        ##To shape Discriminator using refined distribution 
        optimal_batch = self.sess.run(self.optimal_batch, feed_dict={self.z: batch_z, self.inputs: batch_images, self.disc_LR: self.config.learning_rate})
        _ = self.sess.run([self.update_d], feed_dict={ self.inputs: batch_images, self.G: optimal_batch, self.disc_LR: self.config.learning_rate})
      else:
        ##To shape Discriminator using default generator distribution 
        _ = self.sess.run([self.update_d], feed_dict={ self.inputs: batch_images, self.z: batch_z, self.disc_LR: self.config.learning_rate})            

    #Testing
    elif self.mode == "testing":
      pass

    else:
      raise NotImplementedError

    return 

  def discriminator(self, image, is_training=True, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'), train=is_training))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'), train=is_training))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv'), train=is_training))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

      return tf.nn.sigmoid(h4), h4

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      self.z_, self.h0_w, self.h0_b = linear(
          z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

      self.h0 = tf.reshape(
          self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(
          h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b = deconv2d(
          h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(
          h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))

      h4, self.h4_w, self.h4_b = deconv2d(
          h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

      return tf.nn.tanh(h4)

  def noisy_real(self, x, x_hat):
    # out = tf.clip_by_value((x + x_hat*1e-3), clip_value_min = -1.0, clip_value_max = 1.0)
    out = tf.clip_by_value((x*0.9 + x_hat*3e-1), clip_value_min = -1.0, clip_value_max = 1.0)
    # out = tf.clip_by_value(x*0.9 - 1.0 + x_hat * 0.0, clip_value_min = -1.0, clip_value_max = 1.0)
    return out

  def score_refinement(self, batch_z, batch_images, counter, epoch, idx):
    
    ## Get Quantitative Comparisons
    # eval_z = np.random.uniform(-1, 1, [self.eval_size, self.z_dim]).astype(np.float32)
    # eval_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]

    # Visualization for Collaborative Sampling    
    optimal_batch, input_batch, optimal_grad, grad_default, grad_default_real, real_sigmoid, default_sigmoid, optimal_sigmoid, optimal_step, D_sigmoid_real = \
      self.sess.run([self.teacher.optimal_batch, self.G, self.teacher.optimal_grad, self.grad_default, self.grad_default_real, self.teacher.real_sigmoid, self.teacher.default_sigmoid, self.teacher.optimal_sigmoid, self.teacher.optimal_step, self.D], feed_dict={self.z: batch_z, self.inputs: batch_images, self.disc_LR: self.config.learning_rate})
    # print(np.max(np.squeeze(optimal_grad[0] - grad_default[0][0][0])), np.min(np.squeeze(optimal_grad[0] - grad_default[0][0][0])))
    plot_figs(grad_default, default_sigmoid, grad_default_real, D_sigmoid_real, self.eval_size, optimal_batch, input_batch, optimal_grad, counter, epoch, idx, self.config)

    # optimal_eval_batch = []
    # default_eval_batch = []
    # for i in range(10):
    #   eval_z = np.random.uniform(-1, 1, [self.eval_size, self.z_dim]).astype(np.float32)
    #   opt_batch, def_batch = self.sess.run([self.teacher.optimal_batch, self.G], feed_dict={self.z: eval_z, self.inputs: eval_images, self.disc_LR: self.config.learning_rate})
    #   optimal_eval_batch.append(opt_batch)
    #   default_eval_batch.append(def_batch)
    # optimal_eval_batch = np.array(np.concatenate(optimal_eval_batch))
    # default_eval_batch = np.array(np.concatenate(default_eval_batch))

    # # print(optimal_eval_batch.shape)


    # # print(self.data_X.shape)
    # self.inception_score_mnist = self.eval_score.eval({self.evaluator: default_eval_batch})
    # print("Inception of MNIST default: ", self.inception_score_mnist)
    
    # self.incp_score.append(self.inception_score_mnist)
    # # print("Inception List default: ", self.incp_score) 

    # self.inception_score_mnist_refined = self.eval_score.eval({self.evaluator: optimal_eval_batch})
    # print("Inception of MNIST refined: ", self.inception_score_mnist_refined)

    # self.incp_score_refined.append(self.inception_score_mnist_refined)
    # # print("Inception List refined: ", self.incp_score_refined) 

    # # fid_score_mnist = self.fid_eval_score.eval({self.z: eval_z, self.inputs: batch_images})
    # self.fid_score_mnist = self.fid_eval_score.eval({self.evaluator: default_eval_batch, self.eval_real: self.data_X[:640]})              
    # print("FID Score of MNIST default: ", self.fid_score_mnist)

    # self.fid_score.append(self.fid_score_mnist)
    # # print("FID List default: ", self.fid_score)

    # self.fid_score_mnist_refined = self.fid_eval_score.eval({self.evaluator: optimal_eval_batch, self.eval_real: self.data_X[:640]})
    # print("FID Score of MNIST refined: ", self.fid_score_mnist_refined)

    # self.fid_score_refined.append(self.fid_score_mnist_refined)
    # # print("FID List refined: ", self.fid_score_refined)

    return 

  def score_training(self, counter, epoch, idx):

    ## Get Quantitative Comparisons
    # eval_z = np.random.uniform(-1, 1, [self.eval_size, self.z_dim]).astype(np.float32)
    # eval_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
    # eval_samples = self.sess.run(self.G,
    # feed_dict={
    #     self.z: eval_z,
    # })

    # # self.inception_score_mnist = self.eval_score.eval({self.z: eval_z})
    # self.inception_score_mnist = self.eval_score.eval({self.evaluator: eval_samples})
    # print("Inception of MNIST default: ", self.inception_score_mnist)
    
    # self.incp_score.append(self.inception_score_mnist)
    # print("Inception List default: ", incp_score) 

    # # fid_score_mnist = self.fid_eval_score.eval({self.z: eval_z, self.inputs: batch_images})
    # self.fid_score_mnist = self.fid_eval_score.eval({self.evaluator: eval_samples, self.eval_real: self.data_X[:640]})              
    # print("FID Score of MNIST default: ", self.fid_score_mnist)

    # self.fid_score.append(self.fid_score_mnist)
    # # print("FID List default: ", fid_score)

    return

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      h0 = tf.reshape(
          linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
          [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))

      h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))

      h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))

      h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))

      h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

      return tf.nn.tanh(h4)

  def evaluator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      h0 = tf.reshape(
          linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
          [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))

      h1 = deconv2d(h0, [self.eval_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))

      h2 = deconv2d(h1, [self.eval_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))

      h3 = deconv2d(h2, [self.eval_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))

      h4 = deconv2d(h3, [self.eval_size, s_h, s_w, self.c_dim], name='g_h4')

      return tf.nn.tanh(h4)
      
  def load_mnist(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0

    X += 1
    X /= 128
    X -= 1

    return X,y_vec


  def unpickle(self, relpath): 
    print(relpath)  
    with open(relpath, 'rb') as fp:
        d = _pickle.load(fp,encoding='bytes')
    return d

  def prepare_input(self, data=None, labels=None):
    image_height = 32
    image_width = 32
    image_depth = 3
    assert(data.shape[1] == image_height * image_width * image_depth)
    assert(data.shape[0] == labels.shape[0])
    is_nan = np.isnan(data)
    is_inf = np.isinf(data)
    if np.any(is_nan) or np.any(is_inf):
        print('data is not well-formed : is_nan {n}, is_inf: {i}'.format(n= np.any(is_nan), i=np.any(is_inf)))
    #data is transformed from (no_of_samples, 3072) to (no_of_samples , image_height, image_width, image_depth)
    #make sure the type of the data is no.float32
    data = data.reshape([-1,image_depth, image_height, image_width])
    data = data.transpose([0, 2, 3, 1])
    data = data.astype(np.float32)
    #print("prepare_input: ",len(data),len(labels))
    return data, labels


  def read_cifar10(self,filename): # queue one element

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3

    data = self.unpickle(filename)
    #print(data.keys())
    #value = np.asarray(data[b'data']).astype(np.float32)
    #labels = np.asarray(data[b'labels']).astype(np.int32)
    value = np.asarray(data[b'data']).astype(np.float32)
    labels = np.asarray(data[b'labels']).astype(np.int32)
    
    #print("read_cifar10: ",len(value),len(labels))
    return self.prepare_input(value,labels)
    #return prepare_input(value.astype(np.float32),labels.astype(np.int32))

  def load_cifar10(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)

    filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]
    #filenames = ['data_batch_%d.bin' % i for i in xrange(1, 6)]
    filenames.append(os.path.join(data_dir, 'test_batch'))

    for idx , filename in enumerate(filenames):
        temp_X, temp_y = self.read_cifar10(filename)
        print("load_cifar10 for temp shape:",temp_X.shape,temp_y.shape)
        if idx == 0:
            dataX = temp_X
            labely = temp_y
        else:
            dataX = np.append(dataX,temp_X)
            labely = np.append(labely,temp_y)
        dataX = dataX.reshape([-1,32, 32, 3])
        print("load_cifar10 for len:",len(dataX),len(labely))
        print("load_cifar10 for shape:",dataX.shape,labely.shape)


    seed = 547
    np.random.seed(seed)
    np.random.shuffle(dataX)
    np.random.seed(seed)
    np.random.shuffle(labely)

    y_vec = np.zeros((len(labely), 10), dtype=np.float)
    for i, label in enumerate(labely):
        y_vec[i, labely[i]] = 1.0

    dataX += 1
    dataX /= 128
    dataX -= 1
    return dataX, y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "NSGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def restore(self, checkpoint_dir, step):
    model_name = "NSGAN.model"
    ckpt_name = os.path.join(checkpoint_dir, model_name) + "-" + str(step)
    self.saver.restore(self.sess, ckpt_name)
    print(" [*] Success to restore {}".format(ckpt_name))

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

  ##Saving Scores
  def save_scores(self, incp_score, fid_score, config, num_it=1):

    if config.teacher_name == 'default':
      with open('scores/{}/num_it_{}_epoch_{}_teacher_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}_inception_score.pickle'.format(
                 config.dataset, num_it, config.epoch, config.teacher_name, config.D_LR, config.G_LR, config.D_it, config.G_it), 'wb') as handle:
        pickle.dump(incp_score, handle)

      if config.dataset == 'mnist':
        with open('scores/{}/num_it_{}_epoch_{}_teacher_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}_fid_score.pickle'.format(
                   config.dataset, num_it, config.epoch, config.teacher_name, config.D_LR, config.G_LR, config.D_it, config.G_it), 'wb') as handle:
          pickle.dump(fid_score, handle)


    else:
      with open('scores/{}/num_it_{}_epoch_{}_teacher_{}_rollout_method_{}_rollout_steps_{}_rollout_rate_{:06.5f}_inception_score.pickle'.format(
                config.dataset, num_it, config.epoch, config.teacher_name, config.rollout_method, config.rollout_steps, config.rollout_rate), 'wb') as handle:
        pickle.dump(incp_score, handle)

      if config.dataset == 'mnist':
        with open('scores/{}/num_it_{}_epoch_{}_teacher_{}_rollout_method_{}_rollout_steps_{}_rollout_rate_{:06.5f}_fid_score.pickle'.format(
                  config.dataset, num_it, config.epoch, config.teacher_name, config.rollout_method, config.rollout_steps, config.rollout_rate), 'wb') as handle:
          pickle.dump(fid_score, handle)