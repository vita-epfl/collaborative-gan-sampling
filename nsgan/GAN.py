#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *
import utils_mnist

from scipy.special import expit as logistic

import sys
sys.path.append(os.path.join('..', 'sampling'))
from utils_sampling import calibration_diagnostic

MIN_EFFICIENCY = 0.001

class GAN(object):
    model_name = "GAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, eval_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_size = int(eval_size/batch_size)*batch_size

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # train
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist('./data/'+self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

        self.image_dims = [self.input_height, self.input_width, self.c_dim]

    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            logit = linear(net, 1, scope='d_fc4')

            return logit

    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))
            net = deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4')
            out = tf.nn.tanh(net)

            return out

    def input_to_feature(self, z, is_training=False):
        with tf.variable_scope("generator", reuse=True):
            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            return net

    def feature_to_data(self, net, is_training=False):
        with tf.variable_scope("generator", reuse=True):
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))
            net = deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4')
            out = tf.nn.tanh(net)
            return out

    def build_model(self, learning_rate, rollout_steps, rollout_rate):

        self.learning_rate = learning_rate

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='inputs')

        # noises
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images
        D_real_logits = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        self.G = self.generator(self.z, is_training=True, reuse=False)
        D_fake_logits = self.discriminator(self.G, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=g_vars)

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=50)

        """ Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)
        self.fake_logits = self.discriminator(self.fake_images, is_training=True, reuse=True)
        self.fake_sigmoids = tf.nn.sigmoid(self.fake_logits)

        """ Evaluation """
        fname_classify_mnist_graph = './external/classify_mnist_graph_def.pb'
        self.eval_fake = tf.placeholder(tf.float32, [None] + self.image_dims, name='eval_fake')
        self.eval_real = tf.placeholder(tf.float32, [None] + self.image_dims, name='eval_real')
        self.inception_score = utils_mnist.mnist_score(self.eval_fake, fname_classify_mnist_graph)
        self.frechet_distance = utils_mnist.mnist_frechet_distance(self.eval_real, self.eval_fake, fname_classify_mnist_graph)

        """ Sampling """
        from rejector import Rejector
        self.rejector = Rejector()

        from idpsampler import IndependenceSampler
        self.mh_sampler = IndependenceSampler(T=20)

        # Collaborative Sampling
        input_to_feature = self.input_to_feature(self.z)

        from functools import partial
        discriminator_refine = partial(self.discriminator, is_training=True, reuse=True)
        def loss_refine(logits):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))

        from collaborator import Refiner
        refiner = Refiner(rollout_steps=rollout_steps, rollout_rate=rollout_rate)
        refiner.set_env(discriminator_refine, self.feature_to_data, loss_refine)
        self.g_refine_detem = refiner.build_refiner(input_to_feature, self.inputs, mode='deterministic')
        self.g_refine_proba = refiner.build_refiner(input_to_feature, self.inputs, mode='probabilistic')

    def train(self, mode, method):
        # configurations
        self.mode = mode
        self.ckpt_num = -1

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, d_loss = self.sess.run([self.d_optim, self.d_loss],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})

                # update G network
                _, g_loss = self.sess.run([self.g_optim, self.g_loss], feed_dict={self.z: batch_z})

                # display training status
                if counter % 200 == 0:
                    # evaluation
                    self.evaluate(counter, method)
                    # save model
                    self.save(self.checkpoint_dir, counter)
                counter += 1

                # # save training results for every 300 steps
                # if np.mod(counter, 300) == 0:
                #     samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                #     tot_num_samples = min(self.sample_num, self.batch_size)
                #     manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                #     manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                #     save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                #                 './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                #                     epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # show temporal results
            self.visualize_results(epoch)

    def shape(self, mode, method, ckpt_num):
        # configurations
        self.mode = mode
        self.ckpt_num = ckpt_num

        # initialize all variables
        tf.global_variables_initializer().run()

        # load models
        self.load(self.checkpoint_dir, ckpt_num)

        counter = 0

        # shape discriminator
        for epoch in range(3):

            # get batch data
            for idx in range(0, self.num_batches):

                if counter % 500 == 0:
                    # evaluation
                    np.random.seed(2019)
                    self.visualize_refinement(counter)
                    self.evaluate(counter, method)

                # batch data
                batch_real = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # shape D network
                if self.mode == "shape":
                    batch_refine = self.sess.run(self.g_refine_proba, feed_dict={self.z: batch_z, self.inputs: batch_real})
                    _, d_loss = self.sess.run([self.d_optim, self.d_loss], feed_dict={self.G: batch_refine, self.inputs: batch_real})
                elif self.mode == "calibrate":
                    batch_fake = self.sess.run(self.fake_images, feed_dict={self.z: batch_z})
                    _, d_loss = self.sess.run([self.d_optim, self.d_loss], feed_dict={self.G: batch_fake, self.inputs: batch_real})
                else:
                    raise NotImplementedError

                counter += 1

    def evaluate(self, counter, method):

        max_num_propose = self.eval_size / MIN_EFFICIENCY

        num_batch = int(self.eval_size/self.batch_size)
        eval_real = self.data_X[:self.eval_size]
        sigmoid_real = np.empty((self.eval_size, 1))
        for i_batch in range(num_batch):
            sigmoid_real[i_batch*self.batch_size:(i_batch+1)*self.batch_size, :] = self.sess.run(self.fake_sigmoids, feed_dict={self.fake_images: eval_real[i_batch*self.batch_size:(i_batch+1)*self.batch_size, :]})
        eval_z = np.random.uniform(-1, 1, [self.eval_size, self.z_dim]).astype(np.float32)

        if method == "standard" or method == "benchmark":
            eval_standard = np.empty([self.eval_size] + self.image_dims)
            sigmoid_standard = np.empty((self.eval_size, 1))
            for i_batch in range(num_batch):
                batch_z = eval_z[i_batch*self.batch_size:(i_batch+1)*self.batch_size, :]
                batch_samples, batch_sigmoid = self.sess.run([self.fake_images, self.fake_sigmoids], feed_dict={self.z: batch_z})
                eval_standard[i_batch*self.batch_size:(i_batch+1)*self.batch_size, :] = batch_samples
                sigmoid_standard[i_batch*self.batch_size:(i_batch+1)*self.batch_size, :] = batch_sigmoid
            # diagnostic discriminator
            z_dawid, brier_score, ece, mce = calibration_diagnostic(sigmoid_standard, sigmoid_real)
            print("Iter %d: z_calib = %.2f, brier_score = %.2f, ece = %.2f, mce = %.2f" % (counter, z_dawid, brier_score, ece, mce))
            # evaluation
            classifier_score_mnist = self.inception_score.eval({self.eval_fake: eval_standard})
            frechet_distance_mnist = self.frechet_distance.eval({self.eval_fake: eval_standard, self.eval_real: eval_real})
            print("Iter #{:d} (standard): CS = {:.4f}, FD = {:.4f}".format(counter, classifier_score_mnist, frechet_distance_mnist))
            fname = os.path.join(self.log_dir, self.mode + "_standard.txt")
            self.write(fname, self.ckpt_num, counter, classifier_score_mnist, frechet_distance_mnist, 1.0, z_dawid, brier_score, ece, mce)

        if (method == "rejection" or method == "benchmark") and self.mode == "calibrate":
            eval_reject = np.empty([self.eval_size] + self.image_dims)
            cnt_propose = self.eval_size
            self.rejector.set_score_max(np.amax(sigmoid_real))
            eval_reject_base = self.rejector.sampling(eval_standard, sigmoid_standard, shift_percent=100.0)
            cnt_reject = eval_reject_base.shape[0]
            if cnt_reject > 0:
                eval_reject[:cnt_reject] = eval_reject_base
            # fill up evaluation set
            while cnt_reject < self.eval_size:
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                batch_samples, batch_sigmoid = self.sess.run([self.fake_images, self.fake_sigmoids], feed_dict={self.z: batch_z})
                if cnt_propose < max_num_propose:
                    eval_reject_batch = self.rejector.sampling(batch_samples, batch_sigmoid, shift_percent=100.0)
                    cnt_batch = eval_reject_batch.shape[0]
                    if cnt_reject > 0:
                        if cnt_reject + cnt_batch < self.eval_size:
                            eval_reject[cnt_reject:cnt_reject+cnt_batch] = eval_reject_batch
                        else:
                            eval_reject[cnt_reject:self.eval_size] = eval_reject_batch[:self.eval_size-cnt_reject]
                    cnt_reject = cnt_reject + cnt_batch
                else:
                    print("Oops, too inefficient")
                    if cnt_reject + self.batch_size < self.eval_size:
                        eval_reject[cnt_reject:] = batch_samples
                    else:
                        eval_reject[cnt_reject:self.eval_size] = batch_samples[:self.eval_size-cnt_reject]
                    cnt_reject = cnt_reject + self.batch_size
                cnt_propose = cnt_propose + self.batch_size
            # evaluation
            classifier_score_mnist = self.inception_score.eval({self.eval_fake: eval_reject})
            frechet_distance_mnist = self.frechet_distance.eval({self.eval_fake: eval_reject, self.eval_real: eval_real})
            efficiency_mnist = cnt_reject / cnt_propose
            print("Iter #{:d} (rejection): CS = {:.4f}, FD = {:.4f}, Eff = {:.4f}".format(counter, classifier_score_mnist, frechet_distance_mnist, efficiency_mnist))
            fname = os.path.join(self.log_dir, self.mode + "_rejection.txt")
            self.write(fname, self.ckpt_num, counter, classifier_score_mnist, frechet_distance_mnist, efficiency_mnist, z_dawid, brier_score, ece, mce)

        if (method == "hastings" or method == "benchmark") and self.mode == "calibrate":
            eval_mh = np.empty([self.eval_size] + self.image_dims)
            cnt_propose = self.eval_size
            self.mh_sampler.set_score_curr(np.mean(sigmoid_real))
            eval_mh_base = self.mh_sampler.sampling(eval_standard, sigmoid_standard)
            cnt_mh = eval_mh_base.shape[0]
            if cnt_mh > 0:
                eval_mh[:cnt_mh] = eval_mh_base
            # fill up evaluation set
            while cnt_mh < self.eval_size:
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                batch_samples, batch_sigmoid = self.sess.run([self.fake_images, self.fake_sigmoids], feed_dict={self.z: batch_z})
                if cnt_propose < max_num_propose:
                    eval_mh_batch = self.mh_sampler.sampling(batch_samples, batch_sigmoid)
                    cnt_batch = eval_mh_batch.shape[0]
                    if cnt_batch > 0:
                        if cnt_mh + cnt_batch < self.eval_size:
                            eval_mh[cnt_mh:cnt_mh+cnt_batch] = eval_mh_batch
                        else:
                            eval_mh[cnt_mh:self.eval_size] = eval_mh_batch[:self.eval_size-cnt_mh]
                    cnt_mh = cnt_mh + cnt_batch
                else:
                    print("Oops, too inefficient")
                    if cnt_mh + self.batch_size < self.eval_size:
                        eval_mh[cnt_mh:] = batch_samples
                    else:
                        eval_mh[cnt_mh:self.eval_size] = batch_samples[:self.eval_size-cnt_mh]
                    cnt_mh = cnt_mh + self.batch_size
                cnt_propose = cnt_propose + self.batch_size
            classifier_score_mnist = self.inception_score.eval({self.eval_fake: eval_mh})
            frechet_distance_mnist = self.frechet_distance.eval({self.eval_fake: eval_mh, self.eval_real: eval_real})
            efficiency_mnist = cnt_mh / cnt_propose
            print("Iter #{:d} (hastings): CS = {:.4f}, FD = {:.4f}, Eff = {:.4f}".format(counter, classifier_score_mnist, frechet_distance_mnist, efficiency_mnist))
            fname = os.path.join(self.log_dir, self.mode + "_hastings.txt")
            self.write(fname, self.ckpt_num, counter, classifier_score_mnist, frechet_distance_mnist, efficiency_mnist, z_dawid, brier_score, ece, mce)

        if (method == "refinement" or method == "benchmark") and self.mode == "shape":
            # refinement step
            eval_refine = np.empty([self.eval_size] + self.image_dims)
            for i_batch in range(num_batch):
                batch_z = eval_z[i_batch*self.batch_size:(i_batch+1)*self.batch_size, :]
                batch_real = eval_real[i_batch*self.batch_size:(i_batch+1)*self.batch_size, :]
                batch_refine = self.sess.run(self.g_refine_detem, feed_dict={self.z: batch_z, self.inputs: batch_real})
                eval_refine[i_batch*self.batch_size:(i_batch+1)*self.batch_size, :] = batch_refine
            classifier_score_mnist = self.inception_score.eval({self.eval_fake: eval_refine})
            frechet_distance_mnist = self.frechet_distance.eval({self.eval_fake: eval_refine, self.eval_real: eval_real})
            print("Iter #{:d} (refinement): CS = {:.4f}, FD = {:.4f}, Eff = {:.4f}".format(counter, classifier_score_mnist, frechet_distance_mnist, 1.0))
            fname = os.path.join(self.log_dir, self.mode + "_refinement.txt")
            self.write(fname, self.ckpt_num, counter, classifier_score_mnist, frechet_distance_mnist, 1.0, z_dawid, brier_score, ece, mce)

            # rejection step
            eval_collab = np.empty([self.eval_size] + self.image_dims)
            cnt_propose = self.eval_size
            self.mh_sampler.set_score_curr(np.mean(sigmoid_real))
            eval_collab_base = self.mh_sampler.sampling(eval_standard, sigmoid_standard)
            cnt_collab = eval_collab_base.shape[0]
            if cnt_collab > 0:
                eval_collab[:cnt_collab] = eval_collab_base
            # fill up evaluation set
            while cnt_collab < self.eval_size:
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                batch_refine = self.sess.run(self.g_refine_detem, feed_dict={self.z: batch_z, self.inputs: batch_real})
                if cnt_propose < max_num_propose:
                    batch_sigmoid = self.sess.run(self.fake_sigmoids, feed_dict={self.fake_images: batch_refine})
                    eval_collab_batch = self.mh_sampler.sampling(batch_refine, batch_sigmoid)
                    cnt_batch = eval_collab_batch.shape[0]
                    if cnt_batch > 0:
                        if cnt_collab + cnt_batch < self.eval_size:
                            eval_collab[cnt_collab:cnt_collab+cnt_batch] = eval_collab_batch
                        else:
                            eval_collab[cnt_collab:self.eval_size] = eval_collab_batch[:self.eval_size-cnt_collab]
                    cnt_collab = cnt_collab + cnt_batch
                else:
                    print("Oops, too inefficient")
                    if cnt_collab + self.batch_size < self.eval_size:
                        eval_collab[cnt_collab:] = batch_refine
                    else:
                        eval_collab[cnt_collab:self.eval_size] = batch_refine[:self.eval_size-cnt_collab]
                    cnt_collab = cnt_collab + self.batch_size
                cnt_propose = cnt_propose + self.batch_size
            classifier_score_mnist = self.inception_score.eval({self.eval_fake: eval_collab})
            frechet_distance_mnist = self.frechet_distance.eval({self.eval_fake: eval_collab, self.eval_real: eval_real})
            efficiency_mnist = cnt_collab / cnt_propose
            print("Iter #{:d} (collaborate): CS = {:.4f}, FD = {:.4f}, Eff = {:.4f}".format(counter, classifier_score_mnist, frechet_distance_mnist, efficiency_mnist))
            fname = os.path.join(self.log_dir, self.mode + "_collaborate.txt")
            self.write(fname, self.ckpt_num, counter, classifier_score_mnist, frechet_distance_mnist, efficiency_mnist, z_dawid, brier_score, ece, mce)

    def visualize_refinement(self, counter):
        # visualize
        image_frame_dim = int(np.floor(np.sqrt(self.batch_size)))
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        batch_real = self.data_X[:self.batch_size]
        batch_samples = self.sess.run(self.fake_images, feed_dict={self.z: batch_z})
        batch_refine = self.sess.run(self.g_refine_detem, feed_dict={self.z: batch_z, self.inputs: batch_real})
        save_images(batch_samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.mode) + '/ckpt_%03d' % self.ckpt_num + '_standard_%04d' % counter + '.png')
        save_images(batch_refine[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.mode) + '/ckpt_%03d' % self.ckpt_num + '_' + self.mode + '_refinement_%04d' % counter + '.png')

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """
        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model'), global_step=step)

    def load(self, checkpoint_dir, step=None):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        if step:
            ckpt_name = 'model-' + str(step)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, 0
        else:
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

    def write(self, filename, ckpt, iteration, inception_score, frechet_distance, efficiency, z_dawid, brier_score, ece, mce):
        file = open(filename, "a+")
        file.write("%d    %d    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f\r\n"
            % (ckpt, iteration, inception_score, frechet_distance, efficiency, z_dawid, brier_score, ece, mce))
        file.close()
