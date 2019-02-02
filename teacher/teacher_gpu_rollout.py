from __future__ import division
import os 
import numpy as np
import tensorflow as tf
np.random.seed(1234)

from utils import *
from policy_adaptive import * 

class TeacherGPURollout(object):
    """docstring for TeacherGPURollout"""
    def __init__(self, args):
        self.forward_steps = args.rollout_steps
        self.step_size = args.rollout_rate
        self.method = args.rollout_method
        self.args = args
        self.policy = PolicyAdaptive(self.step_size, self.method)
        self.log = False
        # self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))

    def set_env(self, gan, sess):
        self.sess = sess
        self.gan = gan

    def sigmoid_cross_entropy_with_logits(self, x, y):
        try:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        except:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    def compute_real_sigmoid(self, real_batch):
        try:
            real_sigmoid, real_logits = self.gan.discriminator(real_batch, is_training=False, reuse=True)
        except:
            real_sigmoid, real_logits = self.gan.discriminator(real_batch, is_reuse=True)
        real_sigmoid =  tf.reduce_mean(real_sigmoid)
        return real_sigmoid

    def compute_forward_sigmoid_and_grad(self, forward_batch):
        try: 
            forward_sigmoid, forward_logits = self.gan.discriminator(forward_batch, reuse=True)
        except: 
            forward_sigmoid, forward_logits = self.gan.discriminator(forward_batch, is_reuse=True)
        g_forward_loss = tf.reduce_mean(self.sigmoid_cross_entropy_with_logits(forward_logits, tf.ones_like(forward_sigmoid)))
        forward_grad = tf.gradients(g_forward_loss, forward_batch)[0]
        return forward_sigmoid, forward_grad

    def build_teacher(self, fake_batch, fake_sigmoid, fake_grad, real_batch):
        task = 'NSGAN'
        if task == '2DGaussian':
            pass

        else:
            self.real_batch = tf.identity(real_batch)
            self.first_batch = tf.identity(fake_batch)
            self.forward_batch = tf.identity(fake_batch)
            self.optimal_step_init = tf.Variable(tf.zeros([1, self.args.batch_size]), trainable=False)
            self.init_fake_sigmoid = tf.identity(fake_sigmoid)
            self.forward_grad = tf.identity(fake_grad)
            self.optimal_batch = tf.identity(fake_batch)

            # real_batch = tf.Variable(tf.constant(self.data[:self.args.batch_size], dtype=tf.float32), trainable=False)
            # forward_batch = tf.Variable(tf.constant(fake_batch), trainable=False)
            # forward_sigmoid = tf.Variable(tf.constant(fake_sigmoid), trainable=False)
            # forward_grad = tf.Variable(tf.constant(fake_grad), trainable=False)

            # optimal_batch = tf.Variable(tf.constant(fake_batch), trainable=False)
            # optimal_step = tf.Variable(tf.zeros([1, self.args.batch_size]), trainable=False)
            # self.sess.run(tf.variables_initializer([real_batch, forward_batch, forward_sigmoid, forward_grad, optimal_batch, optimal_step]))

            self.real_sigmoid = self.compute_real_sigmoid(self.real_batch)
            self.default_sigmoid = tf.squeeze(self.init_fake_sigmoid)
            self.forward_loss = self.real_sigmoid - tf.squeeze(self.init_fake_sigmoid)
            # optimal search 
            self.optimal_sigmoid = tf.squeeze(self.init_fake_sigmoid)            
            self.optimal_loss = self.real_sigmoid - tf.squeeze(self.init_fake_sigmoid)            
            self.optimal_step = tf.squeeze(self.optimal_step_init)
            # forward_batch = self.forward_batch
            # forward_grad = self.forward_grad
            # optimal_batch = self.optimal_batch
            # recursive forward search 
            for i in range(self.forward_steps):

                # forward update 
                self.forward_batch = self.policy.apply_gradient(self.forward_batch, self.forward_grad, self.forward_loss)
                
                # clip to image value range 
                self.forward_batch = tf.clip_by_value(self.forward_batch, clip_value_min=-1.0, clip_value_max=1.0)

                # tf.print(self.sess.run(self.forward_batch)[:3,0,0,0])
                # self.forward_batch = self.sgd_step(self.forward_batch, self.forward_grad, self.forward_loss)
                # # Add print operation
                # forward_batch = tf.Print(forward_batch, [forward_batch], message="This is a: ")

                # compute current value and next grad
                self.forward_sigmoid, self.forward_grad = self.compute_forward_sigmoid_and_grad(self.forward_batch)
            
                # states
                self.forward_loss = self.real_sigmoid - tf.squeeze(self.forward_sigmoid)

                # comparison
                self.indices_update = tf.less(self.forward_loss, self.optimal_loss)
                # indices_update = (optimal_loss - forward_loss) > 0
                self.optimal_loss = tf.where(self.indices_update, self.forward_loss, self.optimal_loss)
                self.optimal_sigmoid = tf.where(self.indices_update, tf.squeeze(self.forward_sigmoid), self.optimal_sigmoid)
                # optimal_loss[indices_update] = forward_loss[indices_update]
                self.optimal_batch = tf.where(self.indices_update, self.forward_batch, self.optimal_batch)
                # optimal_batch[indices_update,:,:,:] = forward_batch[indices_update,:,:,:]
                self.optimal_step = tf.where(self.indices_update, (i+1)*tf.ones_like(self.optimal_step), self.optimal_step)
                # optimal_step[indices_update] = i+1

            # reset teacher
            self.policy.reset_moving_average()

            self.optimal_grad = (self.first_batch - self.optimal_batch)
            self.optimal_step_fin = tf.clip_by_value(self.optimal_step, clip_value_min=1.0, clip_value_max=10000.0)
            self.optimal_grad = self.optimal_grad / (tf.expand_dims(tf.expand_dims(tf.expand_dims(self.optimal_step_fin, axis=1), axis=2), axis=3))

            # self.optimal_batch = tf.clip_by_value(self.optimal_batch, clip_value_min=-1.0, clip_value_max=1.0)

            return self.optimal_grad, self.optimal_batch
            # self.optimal_batch = optimal_batch
            # TODO: precision 32 vs 64
            # assert np.max(np.abs(fake_grad - optimal_grad)) < 1e-8

