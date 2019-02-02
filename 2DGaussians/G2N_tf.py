from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from Datasets import * 
from Teacher import * 
np.random.seed(1234)
tf.set_random_seed(1234)

import sys
sys.path.append(os.path.join('..', 'collections'))

from ops import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--nhidden', type=int, default=64, help='number of hidden neurons')
    parser.add_argument('--nlayers', type=int, default=6, help='number of hidden layers')
    parser.add_argument('--niters', type=int, default=3001, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=400, help='batch size')
    parser.add_argument('--center', type=float, default=0., help='gradpen center')
    parser.add_argument('--LAMBDA', type=float, default=0., help='gradpen weight')
    parser.add_argument('--alpha', type=float, default=None, help='interpolation weight between reals and fakes')
    parser.add_argument('--lrg', type=float, default=5e-3, help='lr for G')
    parser.add_argument('--lrd', type=float, default=1e-2, help='lr for D')
    parser.add_argument('--dataset', type=str, default='8Gaussians',
                        help='dataset to use: 8Gaussians | 25Gaussians | swissroll | mnist')
    parser.add_argument('--scale', type=float, default=5., help='data scaling')
    parser.add_argument('--loss', type=str, default='gan', help='gan | wgan')
    parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use')
    parser.add_argument('--minibatch_discriminate', action='store_true', help='minibatch_discriminate flag')    
    parser.add_argument('--save_data', action='store_true', help='save data flag')    
    parser.add_argument('--use_teacher', action='store_true', help='teacher flag')

    parser.add_argument('--nfc', type=int, default=64, help='number of hidden neurons')
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--num_epoch', type=int, default=20, help='num_epoch')
    parser.add_argument('--dir_model', type=str, default='ckpt/', help='Directory name to save training model')

    return parser.parse_args()

def plot_loss(prefix, g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list):
    f, ax = plt.subplots(1)
    g_loss_array = np.array(g_loss_list)
    d_loss_array = np.array(d_loss_list)
    d_loss_fake_array = np.array(d_loss_fake_list)
    d_loss_real_array = np.array(d_loss_real_list)
    if len(g_loss_list):
        ax.plot(g_loss_array[:,0], g_loss_array[:,1], color="k", label='g_loss')
    ax.plot(d_loss_array[:,0], d_loss_array[:,1], color="r", label='d_loss')
    ax.plot(d_loss_fake_array[:,0], d_loss_fake_array[:,1], color="g", label='d_loss_fake_array')
    ax.plot(d_loss_real_array[:,0], d_loss_real_array[:,1], color="b", label='d_loss_real_array')
    plt.title('G2N Metrics (2D Gaussians)')
    plt.xlabel('Step')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(prefix + 'metrics.png')

class G2N(object):
    model_name = "G2N"     # name for checkpoint

    def __init__(self, sess, args, data, noise):
        self.sess = sess
        self.nhidden = args.nhidden
        self.nlayers = args.nlayers
        self.niters = args.niters
        self.batch_size = args.batch_size
        self.z_dim = 2
        self.lrg = args.lrg
        self.lrd = args.lrd
        self.data = data
        self.noise = noise
        self.scale = args.scale
        self.save_data = args.save_data
        self.minibatch_discriminate = args.minibatch_discriminate

    def minibatch(self, x, num_kernels=5, kernel_dim=3):
        net = tf.layers.dense(inputs=x, units=num_kernels*kernel_dim, name='minibatch')
        activation = tf.reshape(net, (-1, num_kernels, kernel_dim))
        diffs = tf.expand_dims(activation, 3) - \
            tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat([x, minibatch_features], 1)

    def discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            net = tf.layers.dense(inputs=x, units=self.nhidden, activation=None, name='d_fc1')
            # net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
            net = tf.nn.relu(net, name='d_rl1')
            for i in range(self.nlayers-2):
                net = tf.layers.dense(inputs=net, units=self.nhidden, activation=None, name='d_fc'+str(i+2))
                # net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
                net = tf.nn.relu(net, name='d_rl'+str(i+2))
            if self.minibatch_discriminate:
                net = self.minibatch(net)
            out_logit = tf.layers.dense(inputs=net, units=1, name='d_fc'+str(self.nlayers))
            out = tf.nn.sigmoid(out_logit)
            return out, out_logit

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.layers.dense(inputs=z, units=self.nhidden, activation=None, name='g_fc1')
            net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
            net = tf.nn.relu(net, name='g_rl1')
            for i in range(self.nlayers-2):
                net = tf.layers.dense(inputs=net, units=self.nhidden, activation=None, name='g_fc'+str(i+2))
                net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
                net = tf.nn.relu(net, name='g_rl'+str(i+2))
            out = tf.layers.dense(inputs=net, units=2, activation=None, name='g_fc'+str(self.nlayers))
            return out

    def build_model(self):
        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [None, 2], name='real_placeholder')

        # noises
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z_placeholder')

        """ Loss Function """

        # output of D for real images
        D_real, D_real_logits = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        self.generates = self.generator(self.z, is_training=True, reuse=False)
        D_fake, D_fake_logits = self.discriminator(self.generates, is_training=True, reuse=True)

        # get loss for discriminator
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        ''' Plugin '''
        self.grad_default = tf.gradients(self.g_loss, self.generates)[0]
        # grads_collected = tf.gradients(self.generates, g_vars, self.grad_default)
        
        self.grad_plugin = tf.placeholder(tf.float32, [None,2])
        grads_collected = tf.gradients(self.generates, g_vars, self.grad_plugin)
        grads_and_vars_collected = list(zip(grads_collected, g_vars))

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.GradientDescentOptimizer(self.lrd) \
                      .minimize(self.d_loss, var_list=d_vars)
            # self.d_optim = tf.train.AdamOptimizer(self.lrd) \
            #           .minimize(self.d_loss, var_list=d_vars)

            self.g_optim = tf.train.GradientDescentOptimizer(self.lrg).apply_gradients(grads_and_vars_collected)
            # self.g_optim = tf.train.AdamOptimizer(self.lrg).apply_gradients(grads_and_vars_collected)

        """" Testing """
        # for test
        self.fake_samples = self.generator(self.z, is_training=False, reuse=True)
        
        # self.batch_samples = tf.placeholder(tf.float32, [None, 2], name='batch_placeholder')
        self.fake_sigmoid, self.fake_logit = self.discriminator(self.fake_samples, is_training=False, reuse=True)
        self.fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, labels=tf.ones_like(self.fake_sigmoid)))
        self.fake_saliency = tf.gradients(self.fake_loss, self.fake_samples)[0]
        print("self.fake_samples", self.fake_samples)
        print("self.fake_logits", self.fake_logit)
        print("self.fake_loss", self.fake_loss)
        print("self.fake_saliency", self.fake_saliency)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

        # saver to save model
        self.saver = tf.train.Saver()

    def train(self, prefix, teacher=None, sess_teacher=None):

        # initialize all variables
        tf.global_variables_initializer().run()
        
        # summary writer
        self.writer = tf.summary.FileWriter(prefix)
        g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list = [], [], [], []

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        nticks = 20
        if self.save_data:
            dataset = np.zeros((nticks * nticks * self.niters, 6), dtype=float) 
            print("save dataset shape", dataset.shape)

        # loop for epoch
        start_time = time.time()
        for it in range(self.niters):
            real_batch = self.data.next_batch(self.batch_size)
            noise_batch = self.noise.next_batch(self.batch_size)

            # update D network
            _, d_loss, d_loss_fake, d_loss_real, summary_str = self.sess.run([self.d_optim, self.d_loss, self.d_loss_fake, self.d_loss_real, self.d_sum],
                                feed_dict={self.inputs: real_batch, self.z: noise_batch})
            self.writer.add_summary(summary_str, it)
            d_loss_list.append((it,d_loss))
            d_loss_fake_list.append((it,d_loss_fake))
            d_loss_real_list.append((it,d_loss_real))

            noise_logit, noise_grad = self.sess.run([self.fake_logit, self.grad_default], feed_dict={self.z: noise_batch})

            ''' backprop gradient '''
            if args.use_teacher and teacher and sess_teacher:
                # teacher #
                data_input = np.hstack([np.ones([noise_grad.shape[0],1])*it, noise_batch, noise_logit, noise_grad])
                data_input = data_preprocess(data_input)
                pred_grad = sess_teacher.run(teacher.pred, feed_dict={teacher.inputs: data_input})
                pred_grad = pred_postprocess(pred_grad)
                _, g_loss, summary_str = self.sess.run([self.g_optim, self.g_loss, self.g_sum], feed_dict={self.z: noise_batch, self.grad_plugin: pred_grad})
            else:
                # default #
                _, g_loss, summary_str = self.sess.run([self.g_optim, self.g_loss, self.g_sum], feed_dict={self.z: noise_batch, self.grad_plugin: noise_grad})
            self.writer.add_summary(summary_str, it)
            g_loss_list.append((it,g_loss))

            fake_batch, fake_saliency, fake_logit = self.sess.run([self.fake_samples, self.fake_saliency, self.fake_logit], feed_dict={self.z: noise_batch})
            grid_batch = np.zeros((nticks * nticks, 2))
            step = 2 * self.scale / nticks
            for i in range(nticks):
                for j in range(nticks):
                    grid_batch[i * nticks + j, 0] = -self.scale + i * step
                    grid_batch[i * nticks + j, 1] = -self.scale + j * step
            if self.save_data:
                grid_batch += (np.random.rand(nticks * nticks, 2) - 0.5) * step    

            grid_saliency, grid_logit = self.sess.run([self.fake_saliency, self.fake_logit], feed_dict={self.fake_samples: grid_batch})
            if args.use_teacher and teacher and sess_teacher:
                # teacher #
                data_input = np.hstack([np.ones([noise_grad.shape[0],1])*it, grid_batch, grid_logit, grid_saliency])
                data_input = data_preprocess(data_input)
                grid_pred = sess_teacher.run(teacher.pred, feed_dict={teacher.inputs: data_input})
                grid_saliency = pred_postprocess(pred_grad)

            # create dataset
            if self.save_data:
                # fake_dataset = np.hstack([np.ones_like(fake_logit)*it, fake_batch, fake_logit, fake_saliency])
                grid_dataset = np.hstack([np.ones_like(grid_logit)*it, grid_batch, grid_logit, grid_saliency])
                dataset[it * nticks * nticks : (it+1) * nticks * nticks,:] = grid_dataset

            # display training status
            if it % 100 == 0:
                print("Iter: %d, d_loss: %.8f, g_loss: %.8f" % (it, d_loss, g_loss))

                ax.clear()
                ax.scatter(real_batch[:, 0], real_batch[:, 1], s=2, c='k')
                ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='g', marker='o')
                ax.set_xlim((-self.scale, self.scale))
                ax.set_ylim((-self.scale, self.scale))
                ax.quiver(fake_batch[:, 0], fake_batch[:, 1], -fake_saliency[:, 0], -fake_saliency[:, 1], color='g')
                ax.set_title("Iter #{:d}: g_loss = {:.4f}, d_loss = {:.4f}".format(it, g_loss, d_loss))

                ax.quiver(grid_batch[:, 0], grid_batch[:, 1], -grid_saliency[:, 0], -grid_saliency[:, 1], color='b')

                plt.draw()
                plt.savefig(prefix + 'fig_%05d.png' % it, bbox_inches='tight')
                plt.pause(1e-6)
                # plt.show()
                
        if self.save_data:
            print("dataset[:10,:]",dataset[:5,:])
            print("dataset[:-10,:]",dataset[-5:,:])
            np.savez('dataset.npz', data=dataset)
            # np.savez('dataset_' + str(self.lrg) + '_' + str(self.lrd) + '.npz', data=dataset)
        # self.saver.save(self.sess, self.dir_model + 'gan')
        plot_loss(prefix, g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list)

    def visualize_results(self, epoch):
        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        samples = self.sess.run(self.fake_samples, feed_dict={self.z: z_sample})

"""main"""
def main(args):
    # load gan 
    graph_g2n = tf.Graph()

    # open session
    with tf.Session(graph=graph_g2n) as sess:
        # declare instance for GAN
        noise = NoiseDataset()
        data = ToyDataset(distr=args.dataset, scale=args.scale)
        gan = G2N(sess,args,data,noise)

        # build graph
        gan.build_model()

        # train
        prefix = 'figs/tf_default_lrd_' + str(args.lrd) + '_lrg_' + str(args.lrg) + '/'
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        gan.train(prefix)
        print(" [*] Training finished!")

"""run_teacher"""
def run_teacher(args):
    # load teacher 
    graph_teacher = tf.Graph()
    with tf.Session(graph=graph_teacher) as sess_teacher:
        # launch the graph in a session
        teacher = Teacher(sess_teacher,args)
        teacher.build_model()
        teacher.saver.restore(sess_teacher, args.dir_model + 'teacher')

        # load gan 
        graph_g2n = tf.Graph()
        # open session
        with tf.Session(graph=graph_g2n) as sess:
            # declare instance for GAN
            noise = NoiseDataset()
            data = ToyDataset(distr=args.dataset, scale=args.scale)
            gan = G2N(sess,args,data,noise)

            # build graph
            gan.build_model()

            # train
            prefix = 'figs/tf_teacher_lrd_' + str(args.lrd) + '_lrg_' + str(args.lrg) + '/'
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            gan.train(prefix,teacher,sess_teacher)
            print(" [*] Training finished!")

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)     
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    print(args)

    if args.use_teacher:
        run_teacher(args)
    else:
        main(args)
