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
from utils_image import *
import util_mnist
import util_cifar
# import inc_score
import sys 

sys.path.append(os.path.join('..','sampling'))

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64, eval_size=1000,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir='./data', mode=None,  collab_layer=5, imbalance=False, config=None):

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
    self.config = config

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
    self.collab_layer = collab_layer
    
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
      self.G, self.G_collab = self.generator(self.z)
      # self.G_collab         = self.noisy_real(self.G_collab)
    
    else:
      self.G, self.G_collab = self.generator(self.z, collab_layer=self.collab_layer)
      self.forward_batch    = self.collab_to_data(self.G_collab, collab_layer=self.collab_layer)


    self.D, self.D_logits   = self.discriminator(inputs, reuse=False)
    self.sampler            = self.sampler(self.z)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    # self.evaluator          = self.evaluator(self.z)

    self.evaluator = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, 3], name='ev')
    self.eval_real = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, 3], name='real_ev')

    if self.dataset_name == 'mnist':
      MNIST_CLASSIFIER_FROZEN_GRAPH = './model/classify_mnist_graph_def.pb'
      self.inception_score = util_mnist.mnist_score(self.evaluator, MNIST_CLASSIFIER_FROZEN_GRAPH)
      self.frechet_distance = util_mnist.mnist_frechet_distance(self.eval_real, self.evaluator, MNIST_CLASSIFIER_FROZEN_GRAPH)
      self.inception_prediction = util_mnist.mnist_prediction(self.evaluator, MNIST_CLASSIFIER_FROZEN_GRAPH)
    else:
      self.inception_score = util_cifar.get_inception_scores(self.evaluator)
      self.frechet_distance = util_cifar.get_frechet_inception_distance(self.eval_real, self.evaluator)

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

    # build refiner
    print("[!] refiner_name: ", config.refiner_name)

    if config.refiner_name == 'default':
        self.refiner = None 
    elif config.refiner_name == 'gpurollout':
        from refiner_gpu import Refiner
        self.refiner = Refiner(config)
        self.refiner.set_env(self, self.sess)
        if config.collab_layer == 6:
          self.refiner.set_constraints(vmin=-1.0, vmax=1.0)
    else:
        raise NotImplementedError

    self.saver = tf.train.Saver(max_to_keep=self.config.epoch)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      ###### D Optimizer ######
      # Create D optimizer.
      self.d_optim = tf.train.AdamOptimizer(self.disc_LR*config.D_LR, beta1=config.beta1)
      # Compute the gradients for a list of variables.
      self.grads_d_and_vars = self.d_optim.compute_gradients(self.d_loss, var_list=self.d_vars)
      self.grad_default_real = self.d_optim.compute_gradients(self.d_loss_real, var_list=inputs)
      # Ask the optimizer to apply the capped gradients.
      self.update_d = self.d_optim.apply_gradients(self.grads_d_and_vars) 
      ## Get Saliency Map - refiner 
      self.saliency_map = tf.gradients(self.d_loss, self.inputs)[0]

      ###### G Optimizer ######
      ###### G Optimizer ######
      # Create G optimizer.
      self.g_optim = tf.train.AdamOptimizer(config.learning_rate*config.G_LR, beta1=config.beta1)
      
      # Compute the gradients for a list of variables.
      ## With respect to Generator Weights 
      self.grad_default = self.g_optim.compute_gradients(self.g_loss, var_list=[self.G, self.G_collab, self.g_vars]) 
      ## With Respect to Images given to D 
      grads_collected = tf.gradients(self.G, self.g_vars, self.grad_default[0][0])
      grads_and_vars_collected = list(zip(grads_collected, self.g_vars))      
      self.g_refine = self.g_optim.apply_gradients(grads_and_vars_collected)

      ## Computation Graph depends whether we are refining out of distribution
      if config.refiner_name == 'default':
        self.optimal_grad = self.grad_default[1][0]
        self.optimal_batch = self.G_collab - self.optimal_grad 
      else:
        self.optimal_grad, self.optimal_batch = self.refiner.build_refiner(self.G_collab, self.inputs)
      self.refined_batch = self.collab_to_data(self.optimal_batch, collab_layer=self.collab_layer)

  def train(self, config):

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    if config.mode == "training":
      self.plot_every = 5000
    else:
      self.plot_every = 500

    if config.mode != "training":
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

    if config.mode == "testing":
      self.score_testing()
    elif config.mode == "denoising":
      self.score_denoising()
    else:
      pass 

    for epoch in xrange(config.epoch):

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
          if config.mode == "training" or config.mode == "refinement":
            self.update_Discriminator(batch_images, batch_z, counter)
          elif config.mode == "testing" or config.mode == "denoising":
            pass
          else:
            raise NotImplementedError


        ##Update G according to mode
        for _ in xrange(0, config.G_it):
          if config.mode == "training":
            _, g_loss = self.sess.run([self.g_refine, self.g_loss], feed_dict={self.z: batch_z, self.inputs: batch_images, self.disc_LR: self.config.learning_rate})
          elif config.mode == "refinement":
            g_loss = self.sess.run(self.g_loss, feed_dict={self.z: batch_z, self.inputs: batch_images, self.disc_LR: self.config.learning_rate})
          elif config.mode == "testing" or config.mode == "denoising":
            pass
          else:
            raise NotImplementedError

        ## Evaluation
        # Loss 
        if (idx % self.plot_every) == 0:
          errD_fake = self.d_loss_fake.eval({ self.z: batch_z, self.inputs: batch_images, self.disc_LR: self.config.learning_rate })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images, self.disc_LR: self.config.learning_rate })
          errG = self.g_loss.eval({self.z: batch_z, self.inputs: batch_images, self.disc_LR: self.config.learning_rate })
          print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, config.epoch, idx, batch_idxs,
              time.time() - start_time, errD_fake+errD_real, errG))

        if idx % self.plot_every == 0:    
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
            # print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            plot_fig_samples(samples)
          
          ##Get Quantitive Scores and Visualize Refined Samples 
          if config.mode == "training":
            self.score_training()
          elif config.mode == "refinement":
            self.score_refinement(epoch,idx)
          else:
            pass  


      if config.save_model and epoch > 0:
      # if config.save_model and epoch > 0 and epoch % int(config.epoch/5.0) == 0:
        if config.mode == "training" or config.mode == "refinement":
          print(config.checkpoint_dir)
          self.save(config.checkpoint_dir, epoch)
          print("Saving Model")

    ## Results
    if not config.mode == "denoising":
      print("Final Summary: ")
      print(config)
      
      print("Inception Scores")
      print(self.incp_score)
      if config.dataset == 'mnist':
        print("FID Score ")
        print(self.fid_score)

      print("IS Default: ", self.incp_score)
      if self.incp_score_refined:
        print("IS Refined: ", self.incp_score_refined)

      print("FID Default: ", self.fid_score)
      if self.fid_score_refined:
        print("FID Refined: ", self.fid_score_refined)

    return self.incp_score, self.fid_score

  def update_Discriminator(self, batch_images, batch_z, counter):
    # Update D network
    if self.mode == "training":
      _ = self.sess.run([self.update_d], feed_dict={self.inputs: batch_images, self.z: batch_z, self.disc_LR: self.config.learning_rate})
   
    #Refine D network
    elif self.mode == "refinement":
      if self.use_refined:
        ##To shape Discriminator using refined distribution 
        optimal_batch = self.sess.run(self.refined_batch, feed_dict={self.z: batch_z, self.inputs: batch_images, self.disc_LR: self.config.learning_rate})
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

  def generator(self, z, collab_layer=5, y=None):
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

      out = tf.nn.tanh(h4)
      if collab_layer == 0:
        return out, z
      elif collab_layer == 1:
        return out, h0
      elif collab_layer == 2:
        return out, h1      
      elif collab_layer == 3:
        return out, h2
      elif collab_layer == 4:
        return out, h3
      elif collab_layer == 5:
        return out, h4
      elif collab_layer == 6:
        return out, out
      else:
        raise ValueError

  def noisy_real(self, x):
    out = x*0.9 + (x-1)*0.1
    return out

  def score_refinement(self, epoch, idx):
    
    optimal_eval_batch = []
    default_eval_batch = []
    optimal_sigmoid_batch = []
    default_sigmoid_batch = []
    it = self.eval_size // self.batch_size
    for i in range(it):
      prng = np.random.RandomState(i*11111)
      eval_z = prng.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

      if self.config.dataset == 'mnist' or self.config.dataset == 'cifar10':
        eval_images = self.data_X[:self.batch_size]
      else:  
        batch_files = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for batch_file in batch_files]
        if (self.grayscale):
          eval_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          eval_images = np.array(batch).astype(np.float32)

      opt_batch, def_batch = self.sess.run([self.refined_batch, self.G], feed_dict={self.z: eval_z, self.inputs: eval_images, self.disc_LR: self.config.learning_rate})
      optimal_eval_batch.append(opt_batch)
      default_eval_batch.append(def_batch)
    optimal_eval_batch = np.array(np.concatenate(optimal_eval_batch))
    default_eval_batch = np.array(np.concatenate(default_eval_batch))
    
    if self.config.save_figs:
      plot_figs(default_eval_batch, optimal_eval_batch, epoch, idx, self.config)
      plot_pairs(default_eval_batch, optimal_eval_batch, epoch, idx, self.config)

    self.inception_score_mnist = self.inception_score.eval({self.evaluator: default_eval_batch})
    print("IS default     : {:.4f}".format(self.inception_score_mnist))
    self.incp_score.append(self.inception_score_mnist)

    self.inception_score_mnist_refined = self.inception_score.eval({self.evaluator: optimal_eval_batch})
    print("IS refined [#{:d}]: {:.4f}".format(self.collab_layer, self.inception_score_mnist_refined))
    self.incp_score_refined.append(self.inception_score_mnist_refined)
  
    return 

  def score_testing(self):
    
    epoch, idx = 0, 0

    ## Get Quantitative Comparisons
    eval_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

    if self.config.dataset == 'mnist' or self.config.dataset == 'cifar10':
      eval_images = self.data_X[:self.batch_size]
    else:  
      batch_files = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
      batch = [
          get_image(batch_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for batch_file in batch_files]
      if (self.grayscale):
        eval_images = np.array(batch).astype(np.float32)[:, :, :, None]
      else:
        eval_images = np.array(batch).astype(np.float32)

    # Visualization for Collaborative Sampling    
    optimal_batch, input_batch = self.sess.run([self.refined_batch, self.G], feed_dict={self.z: eval_z, self.inputs: eval_images, self.disc_LR: self.config.learning_rate})
    if self.config.save_figs:
      plot_figs(input_batch, optimal_batch, epoch, idx, self.config)
      plot_pairs(input_batch, optimal_batch, epoch, idx, self.config)

    optimal_eval_batch = []
    default_eval_batch = []
    it = self.eval_size // self.batch_size
    for i in range(it):
      eval_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
      opt_batch, def_batch = self.sess.run([self.refined_batch, self.G], feed_dict={self.z: eval_z, self.inputs: eval_images, self.disc_LR: self.config.learning_rate})
      optimal_eval_batch.append(opt_batch)
      default_eval_batch.append(def_batch)
    optimal_eval_batch = np.array(np.concatenate(optimal_eval_batch))
    default_eval_batch = np.array(np.concatenate(default_eval_batch))

    self.inception_score_mnist = self.inception_score.eval({self.evaluator: default_eval_batch})
    print("IS default     : {:.4f}".format(self.inception_score_mnist))
    self.incp_score.append(self.inception_score_mnist)

    self.inception_score_mnist_refined = self.inception_score.eval({self.evaluator: optimal_eval_batch})
    print("IS refined [#{:d}]: {:.4f}".format(self.collab_layer, self.inception_score_mnist_refined))
    self.incp_score_refined.append(self.inception_score_mnist_refined)
  
    return 

  def score_training(self):

    ## Get Quantitative Comparisons
    if self.dataset_name == 'celebA' or self.dataset_name == 'cifar10':
      default_eval_batch = []
      it = self.eval_size // self.batch_size
      for i in range(it):
        eval_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        def_batch = self.sess.run(self.G, feed_dict={self.z: eval_z})
        default_eval_batch.append(def_batch)
      eval_samples = np.array(np.concatenate(default_eval_batch))

      t0 = time.time()
      self.inception_score_mnist = self.inception_score.eval({self.evaluator: eval_samples})
      t1 = time.time()
      print("IS default: ", self.inception_score_mnist)    
      self.incp_score.append(self.inception_score_mnist)

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
    
    # input characteristics
    np.set_printoptions(precision=2)

    # create imbalance 
    targets = [2,3,5,6,8]
    ratio_imbal = 0.2
    if self.config.imbalance:
      X,y, = create_imbalance(X,y,targets,ratio_imbal)
      print("@ Create imbalanced dataset")

    freq = count_frequency(y)
    print("@ y.shape", y.shape)
    print("@ freq", freq)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0

    X /= 255.
    X *= 2.
    X -= 1.
    return X, y_vec


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
    value = np.asarray(data[b'data']).astype(np.float32)
    labels = np.asarray(data[b'labels']).astype(np.int32)
    
    return self.prepare_input(value,labels)

  def load_cifar10(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)

    filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]
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

    dataX /= 255.
    dataX *= 2.
    dataX -= 1.
    return dataX, y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    if self.config.mode == "training":
      model_name = "DCGAN.model"
    else:
      model_name = "DCGAN-layer" + str(self.collab_layer) +  ".model"
    
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def restore(self, checkpoint_dir, step):
    if self.config.mode == "training":
      model_name = "DCGAN.model"
    else:
      model_name = "DCGAN-layer" + str(self.collab_layer) +  ".model"  
    ckpt_name = os.path.join(checkpoint_dir, model_name) + "-" + str(step)
    print("Model file to restore: ", self.sess, ckpt_name)
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

    if config.refiner_name == 'default':
      with open('scores/{}/num_it_{}_epoch_{}_refiner_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}_inception_score.pickle'.format(
                 config.dataset, num_it, config.epoch, config.refiner_name, config.D_LR, config.G_LR, config.D_it, config.G_it), 'wb') as handle:
        pickle.dump(incp_score, handle)

      if config.dataset == 'mnist':
        with open('scores/{}/num_it_{}_epoch_{}_refiner_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}_fid_score.pickle'.format(
                   config.dataset, num_it, config.epoch, config.refiner_name, config.D_LR, config.G_LR, config.D_it, config.G_it), 'wb') as handle:
          pickle.dump(fid_score, handle)


    else:
      with open('scores/{}/num_it_{}_epoch_{}_refiner_{}_rollout_method_{}_rollout_steps_{}_rollout_rate_{:06.5f}_inception_score.pickle'.format(
                config.dataset, num_it, config.epoch, config.refiner_name, config.rollout_method, config.rollout_steps, config.rollout_rate), 'wb') as handle:
        pickle.dump(incp_score, handle)

      if config.dataset == 'mnist':
        with open('scores/{}/num_it_{}_epoch_{}_refiner_{}_rollout_method_{}_rollout_steps_{}_rollout_rate_{:06.5f}_fid_score.pickle'.format(
                  config.dataset, num_it, config.epoch, config.refiner_name, config.rollout_method, config.rollout_steps, config.rollout_rate), 'wb') as handle:
          pickle.dump(fid_score, handle)

  def collab_to_data(self, z, collab_layer=4, is_training=True, reuse=True):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
     with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      if collab_layer == 0:
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

        out = tf.nn.tanh(h4)

      elif collab_layer == 1:
        self.h1, self.h1_w, self.h1_b = deconv2d(
            z, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        out = tf.nn.tanh(h4)

      elif collab_layer == 2:
        h2, self.h2_w, self.h2_b = deconv2d(
            z, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        out = tf.nn.tanh(h4)

      elif collab_layer == 3:        
        h3, self.h3_w, self.h3_b = deconv2d(
            z, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        out = tf.nn.tanh(h4)

      elif collab_layer == 4:
        h4, self.h4_w, self.h4_b = deconv2d(
            z, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        out = tf.nn.tanh(h4)

      elif collab_layer == 5:
        out = tf.nn.tanh(z)

      elif collab_layer == 6:
        out = z

      else:
        raise ValueError

      return out