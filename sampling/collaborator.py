from __future__ import division
import numpy as np
import tensorflow as tf

from policy import PolicyAdaptive

class Refiner():
    """docstring for Refiner"""
    def __init__(self, rollout_steps, rollout_rate, rollout_method="momentum"):
        self.forward_steps = rollout_steps
        self.optimizer = PolicyAdaptive(rollout_rate, rollout_method)
        self.log = False
        self.vmin = None
        self.vmax = None

    def set_env(self, discriminator, feature_to_data, func_loss):
        self.discriminator = discriminator
        self.feature_to_data = feature_to_data
        self.func_loss = func_loss

    def set_constraints(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        print("set_constraints: self.vmin = {:.2f}, self.vmax = {:.2f}".format(self.vmin, self.vmax))

    def compute_forward_logits_and_grad(self, current_feature):
        # forward and backward pass
        forward_outputs = self.feature_to_data(current_feature)
        forward_logits = self.discriminator(forward_outputs)
        forward_loss = self.func_loss(forward_logits)
        forward_grad = tf.gradients(forward_loss, current_feature)[0]

        # sample-wise logits
        shape = forward_logits.get_shape().as_list()
        dim = np.prod(shape[1:])
        forward_logits_flat = tf.reshape(forward_logits, [-1, dim])
        forward_logit_mean = tf.squeeze(tf.reduce_mean(forward_logits_flat, axis=1))

        return forward_logit_mean, forward_grad

    def build_refiner(self, fake_feature, real_batch, mode='deterministic'):

        ## Real Data Statistics Setup
        self.real_logits = self.discriminator(real_batch)
        self.real_logits_mean = tf.reduce_mean(self.real_logits)

        ## Current Batch for Recursion Setup
        self.current_feature = tf.identity(fake_feature)
        self.current_logit, self.forward_grad = self.compute_forward_logits_and_grad(self.current_feature)

        # Default Output Statistics
        self.default_logit = self.current_logit

        if mode == 'probabilistic':
            size_batch = tf.shape(fake_feature).eval()[0]
            indices_batch = np.random.randint(self.forward_steps+1, size=size_batch)

        self.optimal_feature = tf.identity(fake_feature)
        self.optimal_logit = self.current_logit
        self.optimal_step = tf.ones_like(self.optimal_logit)

        # recursive forward search
        for i in range(self.forward_steps):

            # forward update for activation map
            self.current_feature = self.optimizer.apply_gradient(self.current_feature, self.forward_grad)

            # clip data to the valid range, only needed in data space
            if self.vmin and self.vmax:
                self.current_feature = tf.clip_by_value(self.current_feature, clip_value_min=self.vmin, clip_value_max=self.vmax)

            # discriminator pass and next grad
            self.current_logit, self.forward_grad = self.compute_forward_logits_and_grad(self.current_feature)

            # comparison
            if mode == 'probabilistic':
                indices_update = indices_batch == i
            elif mode == 'deterministic':
                indices_update = tf.greater(self.current_logit, self.optimal_logit)

            self.optimal_logit = tf.where(indices_update, self.current_logit, self.optimal_logit)
            self.optimal_feature = tf.where(indices_update, self.current_feature, self.optimal_feature)
            self.optimal_step = tf.where(indices_update, (i+1)*tf.ones_like(self.optimal_step), self.optimal_step)

        # reset refiner
        self.optimizer.reset_moving_average()

        return self.feature_to_data(self.optimal_feature)
