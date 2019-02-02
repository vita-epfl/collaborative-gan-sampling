from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy
from operator import itemgetter
import sys

np.random.seed(1234)

sys.path.append('.')
sys.path.append(os.path.join('..', '2DGaussians'))

from G2N_tf import G2N
from Datasets import * 
from utils import * 
tf.set_random_seed(1234)

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--nhidden', type=int, default=64, help='number of hidden neurons')
    parser.add_argument('--nlayers', type=int, default=6, help='number of hidden layers')
    parser.add_argument('--niters', type=int, default=501, help='number of iterations')
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
    parser.add_argument('--save_data', action='store_true', help='save data flag')
    parser.add_argument('--use_teacher', action='store_true', help='teacher flag')
    parser.add_argument('--dir_model', type=str, default='ckpt/', help='Directory name to save training model')
    parser.add_argument('--minibatch_discriminate', action='store_true', help='minibatch_discriminate flag')    
    parser.add_argument('--teacher_name', type=str, default='rollout', 
                        help='teacher options: default | scalized | mcts | rollout')
    parser.add_argument('--rollout_rate', type=float, default=1e-3, help='rollout rate')
    parser.add_argument('--rollout_method', type=str, default='ladam')
    parser.add_argument('--rollout_steps', type=int, default=50)    
    parser.add_argument('--res', type=int, default=100, help='Resolution of Board')
    parser.add_argument('--reward_it', type=int, default=8000, help='D reward iteration number')

    return parser.parse_args()

def train_optimal_discriminator(args):

    graph_opt_disc = tf.Graph()
    prefix = 'figs/' + str(args.scale) + '/optimal_discriminator/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # open session
    with tf.Session(graph=graph_opt_disc) as sess:
        # declare instance for GAN
        noise = NoiseDataset()
        data = ToyDataset(distr=args.dataset, scale=args.scale)
        gan = G2N(sess,args,data,noise)

        # build graph
        gan.build_model()

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        # settings
        nticks = 20
        d_loss_list, d_loss_fake_list, d_loss_real_list, norm_grad_list = [], [], [], []

        # loop for epoch
        start_time = time.time()
        for it in range(args.niters):
        	# create data 
            real_batch = data.next_batch(gan.batch_size)
            grid_batch = np.zeros((nticks * nticks, 2))
            step = 2 * args.scale / nticks
            for i in range(nticks):
                for j in range(nticks):
                    grid_batch[i * nticks + j, 0] = -args.scale + i * step
                    grid_batch[i * nticks + j, 1] = -args.scale + j * step
            noisy_batch = grid_batch + (np.random.rand(nticks * nticks, 2) - 0.5) * step

            # update D network
            _, d_loss, d_loss_fake, d_loss_real, summary_str = sess.run([gan.d_optim, gan.d_loss, gan.d_loss_fake, gan.d_loss_real, gan.d_sum],
                                feed_dict={gan.inputs: real_batch, gan.generates: noisy_batch})
            d_loss_list.append((it,d_loss))
            d_loss_fake_list.append((it,d_loss_fake))
            d_loss_real_list.append((it,d_loss_real))

            # compute metrics
            grid_grad, grid_sigmoid = sess.run([gan.fake_saliency, gan.fake_sigmoid], feed_dict={gan.fake_samples: grid_batch})
            norm_grad = np.linalg.norm(grid_grad, axis=1)
            norm_median = np.median(norm_grad)
            norm_grad_list.append((it,norm_median))

            # display training status
            if it % 500 == 0:
                plot_samples(prefix, ax, args.scale, real_batch, grid_batch, grid_grad, grid_sigmoid, it, d_loss, norm_median, np.empty(shape=[0,0]))

            # save model 
            if (it % int(args.niters/5) == 0):
                plot_D_reward(prefix, sess, gan, it, scale=args.scale, res=args.res)
                gan.saver.save(sess, args.dir_model + 'gan', global_step=it)

        # post plot 
        plot_loss(prefix, norm_grad_list, [], d_loss_list, d_loss_fake_list, d_loss_real_list)

def train_optimal_generator(args, idx_ckpt=-1):

    graph_opt_gene = tf.Graph()
    prefix = 'figs/' + str(args.scale) + '/optimal_generator/'

    tf.set_random_seed(1234)
    reward_path = 'figs/' + str(args.scale) + '/optimal_discriminator/fig_%05d.h5' % args.reward_it
    print("[!] teacher_name: ", args.teacher_name)

    # open session
    with tf.Session(graph=graph_opt_gene) as sess:
        # declare instance for GAN
        noise = NoiseDataset()
        data = ToyDataset(distr=args.dataset, scale=args.scale)
        gan = G2N(sess,args,data,noise)

        # build graph
        gan.build_model()

        # build teacher
        print("[!] teacher_name: ", args.teacher_name)

        if args.teacher_name == 'default':
            teacher = [] 
        elif args.teacher_name == 'scalized':
            from teacher_scalized import TeacherScalized
            teacher = TeacherScalized(args)
            print("Scalized Teacher")
        elif args.teacher_name == 'direct':
            from teacher_direct import TeacherDirect
            teacher = TeacherDirect(scale=args.scale)
            print("Direct Teacher")
        elif args.teacher_name == 'mcts':
            from teacher_mcts import TeacherMCTS
            teacher = TeacherMCTS(scale=args.scale, xres=args.res, yres=args.res)
            teacher.init_MCTS()
            print("MCTS Teacher")
        elif args.teacher_name == 'rollout':
            from teacher_rollout import TeacherRollout
            teacher = TeacherRollout(args)
            teacher.set_env(gan, sess, data)
            print("Rollout Teacher")        
        else:
            raise NotImplementedError

        # initialize all variables
        tf.global_variables_initializer().run()

        ckpt = tf.train.get_checkpoint_state(args.dir_model)
        ckpt_path = ckpt.all_model_checkpoint_paths[idx_ckpt]
        if ckpt and ckpt_path:
            ckpt_name = os.path.basename(ckpt_path)
            print(ckpt_name)
            gan.saver.restore(sess, os.path.join(args.dir_model, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))

        prefix += ckpt_name + '/' + args.teacher_name + '/'
        if args.teacher_name == 'rollout':
            prefix += args.rollout_method + '/' + str(args.rollout_rate) + '/'

        if not os.path.exists(prefix):
            os.makedirs(prefix)

        plt.close()
        # graph inputs for visualize training results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # settings
        norm_grad_list, g_loss_list = [], [] 

        # loop for epoch
        start_time = time.time()
        for it in range(args.niters):
            # create data
            real_batch = data.next_batch(args.batch_size)
            noise_batch = noise.next_batch(args.batch_size)

            # fetch G feedback (discriminative value + gradient) 
            noise_sample, noise_grad, noise_sigmoid = sess.run([gan.generates, gan.grad_default, gan.fake_sigmoid], feed_dict={gan.z: noise_batch})


            ####################################################
            ##########       manipulate gradient      ##########

            if teacher:
                # update G network
                teacher_grad = teacher.manipulate_gradient(noise_sample, noise_sigmoid, noise_grad)
                _, g_loss, summary_str = sess.run([gan.g_optim, gan.g_loss, gan.g_sum], feed_dict={gan.z: noise_batch, gan.grad_plugin: teacher_grad})
            else:
                # update G network
                teacher_grad = None
                _, g_loss, summary_str = sess.run([gan.g_optim, gan.g_loss, gan.g_sum], feed_dict={gan.z: noise_batch, gan.grad_plugin: noise_grad})

            ####################################################

            # update G network
            _, g_loss, summary_str = sess.run([gan.g_optim, gan.g_loss, gan.g_sum], feed_dict={gan.z: noise_batch, gan.grad_plugin: noise_grad})
            g_loss_list.append((it,g_loss))

            # display training status
            fake_batch, fake_grad = sess.run([gan.fake_samples, gan.fake_saliency], feed_dict={gan.z: noise_batch})

            # grid visualization 
            ngrids = 21
            grid_batch = np.zeros((ngrids * ngrids, 2))

            xmedian, ymedian = np.median(fake_batch[:, 0]), np.median(fake_batch[:, 1])
            grid_length = max(args.scale/2,max(np.ceil(np.max(fake_batch[:, 0])) - np.floor(np.min(fake_batch[:, 0])), np.ceil(np.max(fake_batch[:, 1])) - np.floor(np.min(fake_batch[:, 1]))))
            step = 2 * args.scale / (ngrids-1)
            for i in range(ngrids):
                for j in range(ngrids):
                    grid_batch[i * ngrids + j, 0] = -args.scale + i * step
                    grid_batch[i * ngrids + j, 1] = -args.scale + j * step

            # xmedian, ymedian = np.median(fake_batch[:, 0]), np.median(fake_batch[:, 1])
            # grid_length = max(args.scale/2,max(np.ceil(np.max(fake_batch[:, 0])) - np.floor(np.min(fake_batch[:, 0])), np.ceil(np.max(fake_batch[:, 1])) - np.floor(np.min(fake_batch[:, 1]))))
            # step = grid_length / (ngrids-1)
            # for i in range(ngrids):
            #     for j in range(ngrids):
            #         grid_batch[i * ngrids + j, 0] = xmedian - grid_length / 2.0 + i * step
            #         grid_batch[i * ngrids + j, 1] = ymedian - grid_length / 2.0 + j * step

            # compute metrics
            grid_grad_default, grid_sigmoid = sess.run([gan.fake_saliency, gan.fake_sigmoid], feed_dict={gan.fake_samples: grid_batch})
            if teacher:
                grid_grad_teacher = teacher.manipulate_gradient(grid_batch, grid_sigmoid, grid_grad_default)
            else:
                grid_grad_teacher = None

            norm_grad = np.linalg.norm(grid_grad_default, axis=1)
            norm_median = np.median(norm_grad)
            norm_grad_list.append((it,norm_median))

            norm_noise_grad = np.linalg.norm(noise_grad, axis=1)
            norm_noise_grad_mean = np.mean(norm_noise_grad)

            # display training status
            if it % 10 == 0:
                plot_samples(prefix, ax1, args.scale, real_batch, grid_batch, grid_grad_default, grid_sigmoid, it, g_loss, norm_noise_grad_mean, noise_sample, xmedian - grid_length / 2.0, xmedian + grid_length / 2.0, ymedian - grid_length / 2.0, ymedian + grid_length / 2.0, grid_grad_teacher)
                plot_norm_distribution(ax2, 1 - noise_sigmoid, noise_grad, teacher_grad, args.teacher_name, args.rollout_method, args.rollout_steps, args.rollout_rate)
                fig.savefig(prefix + 'fig_%05d.png' % it, bbox_inches='tight')

        plot_loss(prefix, norm_grad_list, g_loss_list, [], [], [])


if __name__ == '__main__':
    # print settings
    np.set_printoptions(precision=4, suppress=True)
    
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    else:
        print(args)

    # main functions
    # train_optimal_discriminator(args)
    train_optimal_generator(args, 2)
