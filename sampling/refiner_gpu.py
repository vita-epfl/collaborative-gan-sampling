from __future__ import division
import os 
import numpy as np
import tensorflow as tf
np.random.seed(1234)

from utils import *
from policy import * 

class Refiner(object):
    """docstring for Refiner"""
    def __init__(self, args, wgan=False):
        self.forward_steps = args.rollout_steps
        self.step_size = args.rollout_rate
        self.method = args.rollout_method
        self.args = args
        self.policy = PolicyAdaptive(self.step_size, self.method)
        self.log = False
        self.vmin = None
        self.vmax = None
        self.collab_layer = args.collab_layer
        self.wgan = wgan
        # self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))

    def set_env(self, gan, sess):
        self.sess = sess
        self.gan = gan

    def set_constraints(self, vmin, vmax):
        self.vmin = vmin 
        self.vmax = vmax 
        print("set_constraints: self.vmin = {:.2f}, self.vmax = {:.2f}".format(self.vmin, self.vmax))

    def sigmoid_cross_entropy_with_logits(self, x, y):
        try:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        except:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    def compute_sigmoid_and_logit(self, batch):
        try:
            batch_sigmoid, batch_logits = self.gan.discriminator(batch, is_training=False, reuse=True)
        except:
            batch_sigmoid, batch_logits = self.gan.discriminator(batch, is_reuse=True)
        return batch_sigmoid, batch_logits

    def compute_forward_sigmoid_logits_and_grad(self, forward_batch):

        forward_x_batch = self.gan.collab_to_data(forward_batch, collab_layer=self.collab_layer)
        try: 
            forward_sigmoid, forward_logits = self.gan.discriminator(forward_x_batch, is_training=False, reuse=True)
        except: 
            forward_sigmoid, forward_logits = self.gan.discriminator(forward_x_batch, is_reuse=True)

        if not self.wgan:
            g_forward_loss = tf.reduce_mean(self.sigmoid_cross_entropy_with_logits(forward_logits, tf.ones_like(forward_sigmoid)))
        else:
            g_forward_loss = -tf.reduce_mean(forward_logits)
            
        forward_grad = tf.gradients(g_forward_loss, forward_batch)[0]
            
        return forward_sigmoid, forward_logits, forward_grad

    def build_refiner(self, fake_batch, real_batch):
        task = 'NSGAN'
        if task == '2DGaussian':
            pass

        else:
            ## Real Data Statistics Setup
            self.real_batch = tf.identity(real_batch)
            self.real_sigmoid, self.real_logits = self.compute_sigmoid_and_logit(self.real_batch)
            self.real_sigmoid_mean = tf.reduce_mean(self.real_sigmoid)
            self.real_logits_mean = tf.reduce_mean(self.real_logits)
            if self.wgan:
                self.real_comparison = self.real_logits_mean
            else:
                self.real_comparison = self.real_sigmoid_mean

            ## Forward Batch for Recursion Setup
            self.first_batch = tf.identity(fake_batch)
            self.forward_batch = tf.identity(fake_batch)
            self.forward_sigmoid, self.forward_logits, self.forward_grad = self.compute_forward_sigmoid_logits_and_grad(self.forward_batch)
            self.forward_sigmoid_mean = tf.reduce_mean(self.forward_sigmoid)
            self.forward_logits_mean = tf.reduce_mean(self.forward_logits)
            if self.wgan:
                self.init_forward_comparison = self.forward_logits
            else:
                self.init_forward_comparison = self.forward_sigmoid 
            self.forward_loss = self.real_comparison - tf.squeeze(self.init_forward_comparison)

            # Optimal Batch Statistics
            self.optimal_batch = tf.identity(fake_batch)
            self.optimal_step_init = tf.Variable(tf.zeros([1, self.args.batch_size]), trainable=False)        
            self.optimal_loss = self.real_comparison - tf.squeeze(self.init_forward_comparison)            
            self.optimal_step = tf.squeeze(self.optimal_step_init)

            # recursive forward search 
            for i in range(self.forward_steps):

                # forward update 
                self.forward_batch = self.policy.apply_gradient(self.forward_batch, self.forward_grad, self.forward_loss)
                
                # clip to image value range 
                if self.vmin and self.vmax: 
                    self.forward_batch = tf.clip_by_value(self.forward_batch, clip_value_min=self.vmin, clip_value_max=self.vmax)

                # compute current value and next grad
                self.forward_sigmoid, self.forward_logits, self.forward_grad = self.compute_forward_sigmoid_logits_and_grad(self.forward_batch)
            
                # states
                if self.wgan:
                    self.forward_loss = self.real_comparison - tf.squeeze(self.forward_logits)
                else:
                    self.forward_loss = self.real_comparison - tf.squeeze(self.forward_sigmoid)

                # comparison
                self.indices_update = tf.less(self.forward_loss, self.optimal_loss)
                # indices_update = (optimal_loss - forward_loss) > 0
                self.optimal_loss = tf.where(self.indices_update, self.forward_loss, self.optimal_loss)
                # optimal_loss[indices_update] = forward_loss[indices_update]
                self.optimal_batch = tf.where(self.indices_update, self.forward_batch, self.optimal_batch)
                # optimal_batch[indices_update,:,:,:] = forward_batch[indices_update,:,:,:]
                self.optimal_step = tf.where(self.indices_update, (i+1)*tf.ones_like(self.optimal_step), self.optimal_step)
                # optimal_step[indices_update] = i+1

            # reset refiner
            self.policy.reset_moving_average()

            self.optimal_grad = (self.first_batch - self.optimal_batch)
            self.optimal_step_fin = tf.clip_by_value(self.optimal_step, clip_value_min=1.0, clip_value_max=10000.0)
            self.optimal_grad = self.optimal_grad / (tf.expand_dims(tf.expand_dims(tf.expand_dims(self.optimal_step_fin, axis=1), axis=2), axis=3))

            # print(self.optimal_step)
            return self.optimal_grad, self.optimal_batch



            # self.optimal_batch = optimal_batch
            # TODO: precision 32 vs 64
            # assert np.max(np.abs(fake_grad - optimal_grad)) < 1e-8


                # tf.print(self.sess.run(self.forward_batch)[:3,0,0,0])
                # self.forward_batch = self.sgd_step(self.forward_batch, self.forward_grad, self.forward_loss)
                # # Add print operation
                # forward_batch = tf.Print(forward_batch, [forward_batch], message="This is a: ")

            # self.init_fake_sigmoid = tf.identity(fake_sigmoid)
            # self.forward_grad = tf.identity(fake_grad)
