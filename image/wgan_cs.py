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
import sys 

sys.path.append(os.path.join('..','sampling'))

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


class WGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64, eval_size=12800,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir='./data', mode=None, collab_layer=4, lambd=0.10, config=None):

    self.config = config

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

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.data_dir = data_dir
    self.mode = config.mode

    self.data_X, self.data_y = self.load_mnist()
    self.c_dim = self.data_X[0].shape[-1]

    self.grayscale = (self.c_dim == 1)
    self.use_refined = config.use_refined
    self.collab_layer = collab_layer
    self.lambd = lambd

    self.build_model(config)

  def build_model(self, config):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    self.disc_LR = tf.placeholder(tf.float32, (), name='disc_LR')

    image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')

    #Denoising Setup
    if config.denoise:
      print("Denoising Experiment")
      G, _                  = self.generator(self.z)
      self.G                = self.noisy_real(inputs,G)
    
    else:
      self.G, self.G_collab = self.generator(self.z, collab_layer=self.collab_layer)
      self.forward_batch    = self.collab_to_data(self.G_collab, collab_layer=self.collab_layer)

    self.D, self.D_logits   = self.discriminator(inputs, reuse=False)
    self.sampler            = self.sampler(self.z)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    # self.evaluator          = self.evaluator(self.z)
    self.evaluator = tf.placeholder(tf.float32, [None, 28, 28, 1], name='ev')
    self.eval_real = tf.placeholder(tf.float32, [None, 28, 28, 1], name='real_ev')

    if self.dataset_name == 'mnist':
      MNIST_CLASSIFIER_FROZEN_GRAPH = './model/classify_mnist_graph_def.pb'
      self.inception_score = util_mnist.mnist_score(self.evaluator, MNIST_CLASSIFIER_FROZEN_GRAPH)
      self.frechet_distance = util_mnist.mnist_frechet_distance(self.eval_real, self.evaluator, MNIST_CLASSIFIER_FROZEN_GRAPH)
      self.inception_prediction = util_mnist.mnist_prediction(self.evaluator, MNIST_CLASSIFIER_FROZEN_GRAPH)

    # get loss for discriminator
    self.d_loss_real = - tf.reduce_mean(self.D_logits)
    self.d_loss_fake = tf.reduce_mean(self.D_logits_)
    self.d_loss      = self.d_loss_real + self.d_loss_fake

    # get loss for generator
    self.g_loss = - self.d_loss_fake


    """ Gradient Penalty """
    # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
    alpha            = tf.random_uniform(shape=self.inputs.get_shape(), minval=0.,maxval=1.)
    differences      = self.G - self.inputs # This is different from MAGAN
    interpolates     = self.inputs + (alpha * differences)
    _, D_inter       = self.discriminator(interpolates, is_training=True, reuse=True)
    gradients        = tf.gradients(D_inter, [interpolates])[0]
    slopes           = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    self.d_loss     += self.lambd * gradient_penalty


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
        if config.collab_layer == 5:
          self.refiner.set_constraints(vmin=-1.0, vmax=1.0)
    else:
        raise NotImplementedError

    self.saver = tf.train.Saver(max_to_keep=20)
    
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
      # Create G optimizer.
      self.g_optim = tf.train.AdamOptimizer(config.learning_rate*config.G_LR, beta1=config.beta1)
      
      # Compute the gradients for a list of variables.
      ## With respect to Generator Weights - AutoLoss
      self.grad_default = self.g_optim.compute_gradients(self.g_loss, var_list=[self.G, self.G_collab, self.g_vars]) 
      ## With Respect to Images given to D - refiner'
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
      self.plot_every = 500
    else:
      self.plot_every = 500

    if config.mode != "training":
      self.restore(config.load_model_dir, config.load_epoch)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))  
    sample_inputs = self.data_X[0:self.sample_num]

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

      ##Load Batchsize
      if config.mode == 'denoising':
        batch_idxs = 1
      else:
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      
      ##Batch GD
      for idx in xrange(0, int(batch_idxs)):

        counter += 1

        ##Load Data
        batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
        batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]

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

        # Image
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
              plot_fig_samples(samples)
          
          ##Get Quantitive Scores and Visualize Refined Samples 
          if config.mode == "training":
            self.score_training()
          elif config.mode == "refinement" or config.mode == "testing":
            self.score_refinement(epoch,idx)
          elif config.mode == "denoising":
            self.score_denoising()
          else:
            raise NotImplementedError

        if config.save_model and (counter % self.plot_every) == 0:
          if config.mode == "training":
            print(config.checkpoint_dir)
            self.save(config.checkpoint_dir, epoch)
            print("Saving Model")

    if config.save_model:
      if config.mode == "refinement":
        print(config.checkpoint_dir)
        self.save(config.checkpoint_dir, epoch)
        print("Saving Model")

    ## Results
    if not config.mode == "denoising":
      print("Final Summary: ")
      print(config)

      print("IS Default: ", self.incp_score)
      if self.incp_score_refined:
        print("IS Refined: ", self.incp_score_refined)
        print("IS Top (loaded iter {:d}): default = {:.4f}, refined = {:.4f}".format(self.config.load_epoch, self.average_top(self.incp_score), self.average_top(self.incp_score_refined) ))

      print("FID Default: ", self.fid_score)
      if self.fid_score_refined:
        print("FID Refined: ", self.fid_score_refined)
        print("FID Top (loaded iter {:d}): default = {:.4f}, refined = {:.4f}".format(self.config.load_epoch, self.average_top(self.fid_score,'low'), self.average_top(self.fid_score_refined,'low') ))

    return self.incp_score, self.fid_score

  def average_top(self, data, rule='high', num=3):
    arr = np.array(data)
    if rule is not 'high':
      arr = arr[::-1]
    order = np.argsort(arr)
    return np.mean(arr[order<num])

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

  def discriminator(self, x, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    with tf.variable_scope("discriminator", reuse=reuse):

      net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
      net = lrelu(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'))
      # net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
      net = tf.reshape(net, [self.batch_size, -1])
      net = lrelu(linear(net, 1024, scope='d_fc3'))
      # net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
      out_logit = linear(net, 1, scope='d_fc4')
      out = tf.nn.sigmoid(out_logit)

      return out, out_logit

  def generator(self, z, collab_layer=4, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.variable_scope("generator", reuse=reuse):
      layer1 = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
      layer2 = tf.nn.relu(bn(linear(layer1, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
      layer2 = tf.reshape(layer2, [self.batch_size, 7, 7, 128])
      layer3 = tf.nn.relu(
          bn(deconv2d(layer2, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
             scope='g_bn3'))
      layer4 = deconv2d(layer3, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4')
      out = tf.nn.tanh(layer4)
      if collab_layer == 0:
        return out, z
      elif collab_layer == 1:
        return out, layer1
      elif collab_layer == 2:
        return out, layer2      
      elif collab_layer == 3:
        return out, layer3
      elif collab_layer == 4:
        return out, layer4
      elif collab_layer == 5:
        return out, out
      else:
        raise ValueError


  def noisy_real(self, x, x_hat):
    # out = tf.clip_by_value((x + x_hat*1e-3), clip_value_min = -1.0, clip_value_max = 1.0)
    out = tf.clip_by_value((x*0.9 + x_hat*3e-1), clip_value_min = -1.0, clip_value_max = 1.0)
    # out = tf.clip_by_value(x*0.9 - 1.0 + x_hat * 0.0, clip_value_min = -1.0, clip_value_max = 1.0)
    return out

  def score_training(self):

    ## Get Quantitative Comparisons
    default_eval_batch = []
    it = self.eval_size // self.batch_size
    for i in range(it):
      eval_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
      def_batch = self.sess.run(self.G, feed_dict={self.z: eval_z})
      default_eval_batch.append(def_batch)
    eval_samples = np.array(np.concatenate(default_eval_batch))

    self.inception_score_mnist = self.inception_score.eval({self.evaluator: eval_samples})
    print("Inception of MNIST default: ", self.inception_score_mnist)    
    self.incp_score.append(self.inception_score_mnist)

    self.fid_score_mnist = self.frechet_distance.eval({self.evaluator: eval_samples, self.eval_real: self.data_X[:self.eval_size]})              
    print("FID Score of MNIST default: ", self.fid_score_mnist)
    self.fid_score.append(self.fid_score_mnist)

    return

  def score_refinement(self, epoch, idx):
    
    ## Get Quantitative Comparisons
    eval_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
    eval_images = self.data_X[:self.batch_size]

    # Visualization for Collaborative Sampling    
    # optimal_batch, input_batch, optimal_grad, grad_default, grad_default_real, real_sigmoid, default_sigmoid, optimal_sigmoid, optimal_step, D_sigmoid_real = \
      # self.sess.run([self.refined_batch, self.G, self.refiner.optimal_grad, self.grad_default, self.grad_default_real, self.refiner.real_sigmoid_mean, self.refiner.default_sigmoid, self.refiner.optimal_sigmoid, self.refiner.optimal_step, self.D], feed_dict={self.z: eval_z, self.inputs: eval_images, self.disc_LR: self.config.learning_rate})

    # Visualization for Collaborative Sampling    
    optimal_batch, input_batch = self.sess.run([self.refined_batch, self.G], feed_dict={self.z: eval_z, self.inputs: eval_images, self.disc_LR: self.config.learning_rate})
    plot_figs(input_batch, optimal_batch, epoch, idx, self.config)

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
    print("IS MNIST default: ", self.inception_score_mnist)
    self.incp_score.append(self.inception_score_mnist)

    self.inception_score_mnist_refined = self.inception_score.eval({self.evaluator: optimal_eval_batch})
    print("IS MNIST refined at layer #[{:d}]: {:f}".format(self.collab_layer, self.inception_score_mnist_refined))
    self.incp_score_refined.append(self.inception_score_mnist_refined)

    self.fid_score_mnist = self.frechet_distance.eval({self.evaluator: default_eval_batch, self.eval_real: self.data_X[:self.eval_size]})      
    print("FID MNIST default: ", self.fid_score_mnist)
    self.fid_score.append(self.fid_score_mnist)

    self.fid_score_mnist_refined = self.frechet_distance.eval({self.evaluator: optimal_eval_batch, self.eval_real: self.data_X[:self.eval_size]})
    print("FID MNIST refined at layer #[{:d}]: {:f}".format(self.collab_layer, self.fid_score_mnist_refined))
    self.fid_score_refined.append(self.fid_score_mnist_refined)

    return 

  def score_testing(self):

    epoch, idx = 0, 0
    
    ## Get Quantitative Comparisons
    eval_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
    eval_images = self.data_X[:self.batch_size]

    # Visualization for Collaborative Sampling    
    optimal_batch, input_batch = self.sess.run([self.refined_batch, self.G], feed_dict={self.z: eval_z, self.inputs: eval_images, self.disc_LR: self.config.learning_rate})
    plot_figs(input_batch, optimal_batch, epoch, idx, self.config)

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
    print("IS MNIST default: ", self.inception_score_mnist)
    self.incp_score.append(self.inception_score_mnist)

    self.inception_score_mnist_refined = self.inception_score.eval({self.evaluator: optimal_eval_batch})
    print("IS MNIST refined at layer #[{:d}]: {:f}".format(self.collab_layer, self.inception_score_mnist_refined))
    self.incp_score_refined.append(self.inception_score_mnist_refined)

    self.fid_score_mnist = self.frechet_distance.eval({self.evaluator: default_eval_batch, self.eval_real: self.data_X[:self.eval_size]})      
    print("FID MNIST default: ", self.fid_score_mnist)
    self.fid_score.append(self.fid_score_mnist)

    self.fid_score_mnist_refined = self.frechet_distance.eval({self.evaluator: optimal_eval_batch, self.eval_real: self.data_X[:self.eval_size]})
    print("FID MNIST refined at layer #[{:d}]: {:f}".format(self.collab_layer, self.fid_score_mnist_refined))
    self.fid_score_refined.append(self.fid_score_mnist_refined)

    plot_figs(input_batch, optimal_batch, 0, 0, self.config)

    prefix = "real"
    foldername = "./indi_figs/real"
    plot_single_fig(input_batch, prefix, foldername)

    prefix = "epoch_" + str(self.config.load_epoch) + \
            "_layer_" + str(self.config.collab_layer) + \
            "_step_" + str(self.config.rollout_steps) + \
            "_rate_" + str(self.config.rollout_rate)
    foldername = "./indi_figs/layer_" + str(self.config.collab_layer) + \
                  "_step_" + str(self.config.rollout_steps) + \
                  "_rate_" + str(self.config.rollout_rate)
    plot_single_fig(optimal_batch, prefix, foldername)

    prefix = "epoch_" + str(self.config.load_epoch) + \
            "_layer_" + str(self.config.collab_layer) + \
            "_step_" + str(self.config.rollout_steps) + \
            "_rate_" + str(self.config.rollout_rate) + "_diff"
    foldername = "./indi_figs/layer_" + str(self.config.collab_layer) + \
                  "_step_" + str(self.config.rollout_steps) + \
                  "_rate_" + str(self.config.rollout_rate)
    plot_single_fig((optimal_batch - input_batch)/2.0, prefix, foldername, vmin=0.3, vmax=0.7)

    mat = input_batch - optimal_batch 
    flat = np.reshape(mat,(np.size(mat,0),-1))

    norm = np.mean(np.linalg.norm(flat,axis=1))
    norm_inf = np.mean(np.linalg.norm(flat,np.inf,axis=1))
    norm_1 = np.mean(np.linalg.norm(flat,1,axis=1))
    
    print("Distance #{:d}: L2 = {:.4f}, L1 = {:.4f}, L-inf = {:.4f}".format(self.config.collab_layer, norm, norm_1, norm_inf))

    return 

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=False, scope='g_bn1'))
      net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=False, scope='g_bn2'))
      net = tf.reshape(net, [self.batch_size, 7, 7, 128])
      net = tf.nn.relu(
          bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=False,
             scope='g_bn3'))

      out = tf.nn.tanh(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

      return out
      
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

  @property
  def model_dir(self):
    if self.config.imbalance:
      return "{}_{}_{}_{}_imbalance".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
    else:
      return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    if self.config.mode == "training":
      model_name = "WGAN-train.model"
    else:
      model_name = "WGAN-layer" + str(self.collab_layer) +  ".model"

    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def restore(self, checkpoint_dir, step):
    if self.config.mode == "training" or self.config.mode == "refinement":
      model_name = "WGAN-train.model"
    else:
      model_name = "WGAN-layer" + str(self.collab_layer) +  ".model"

    ckpt_name = os.path.join(checkpoint_dir, self.model_dir, model_name) + "-" + str(step)
    self.saver.restore(self.sess, ckpt_name)
    print(" [*] Success to restore {}".format(ckpt_name))

  # def load(self, checkpoint_dir):
  #   import re
  #   print(" [*] Reading checkpoints...")
  #   checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

  #   ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  #   if ckpt and ckpt.model_checkpoint_path:
  #     ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
  #     self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
  #     counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
  #     print(" [*] Success to read {}".format(ckpt_name))
  #     return True, counter
  #   else:
  #     print(" [*] Failed to find a checkpoint")
  #     return False, 0

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
      with open('scores/{}/num_it_{}_epoch_{}_refiner_{}_rollout_method_{}_rollout_steps_{}_rollout_rate_{:.5f}_inception_score.pickle'.format(
                config.dataset, num_it, config.epoch, config.refiner_name, config.rollout_method, config.rollout_steps, config.rollout_rate), 'wb') as handle:
        pickle.dump(incp_score, handle)

      if config.dataset == 'mnist':
        with open('scores/{}/num_it_{}_epoch_{}_refiner_{}_rollout_method_{}_rollout_steps_{}_rollout_rate_{:.5f}_fid_score.pickle'.format(
                  config.dataset, num_it, config.epoch, config.refiner_name, config.rollout_method, config.rollout_steps, config.rollout_rate), 'wb') as handle:
          pickle.dump(fid_score, handle)

  def collab_to_data(self, z, collab_layer=4, is_training=True, reuse=True):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.variable_scope("generator", reuse=reuse):

      if collab_layer == 0:
        layer1 = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
        layer2 = tf.nn.relu(bn(linear(layer1, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
        layer2 = tf.reshape(layer2, [self.batch_size, 7, 7, 128])
        layer3 = tf.nn.relu(
            bn(deconv2d(layer2, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
               scope='g_bn3'))
        layer4 = deconv2d(layer3, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4')
        out = tf.nn.tanh(layer4)

      elif collab_layer == 1:
        layer2 = tf.nn.relu(bn(linear(z, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
        layer2 = tf.reshape(layer2, [self.batch_size, 7, 7, 128])
        layer3 = tf.nn.relu(
            bn(deconv2d(layer2, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
               scope='g_bn3'))
        layer4 = deconv2d(layer3, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4')
        out = tf.nn.tanh(layer4)

      elif collab_layer == 2:
        layer3 = tf.nn.relu(
            bn(deconv2d(z, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
               scope='g_bn3'))
        layer4 = deconv2d(layer3, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4')
        out = tf.nn.tanh(layer4) 

      elif collab_layer == 3:        
        layer4 = deconv2d(z, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4')
        out = tf.nn.tanh(layer4) 

      elif collab_layer == 4:
        out = tf.nn.tanh(z)

      elif collab_layer == 5:
        out = z

      else:
        raise ValueError

      return out
