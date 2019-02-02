from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
import shutil
import time
import torch
from torchvision import transforms, datasets

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
    parser.add_argument('--niters', type=int, default=20001, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=400, help='batch size')
    parser.add_argument('--center', type=float, default=0., help='gradpen center')
    parser.add_argument('--LAMBDA', type=float, default=0., help='gradpen weight')
    parser.add_argument('--alpha', type=float, default=None, help='interpolation weight between reals and fakes')
    parser.add_argument('--lrg', type=float, default=5e-3, help='lr for G')
    parser.add_argument('--lrd', type=float, default=1e-2, help='lr for D')
    parser.add_argument('--dataset', type=str, default='Imbal-8Gaussians',
                        help='dataset to use: 8Gaussians | 25Gaussians | swissroll | mnist')
    parser.add_argument('--scale', type=float, default=10., help='data scaling')
    parser.add_argument('--loss', type=str, default='gan', help='gan | wgan')
    parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use')
    parser.add_argument('--save_data', action='store_true', help='save data flag')
    parser.add_argument('--save_model', action='store_true', help='save data flag')
    parser.add_argument('--dir_model', type=str, default='ckpt/', help='Directory name to save training model')
    parser.add_argument('--minibatch_discriminate', action='store_true', help='minibatch_discriminate flag')    
    parser.add_argument('--nrefines', type=int, default=0, help='number of refinements')
    parser.add_argument('--ckpt_num', type=int, default=2000, help='ckpt number')
    parser.add_argument('--plot_every', type=int, default=1000, help='ckpt number')


    parser.add_argument('--mode', type=str, default='training',
                        help='type of running: training, refinement, testing')
    parser.add_argument('--refine_method', type=str, default='freeze_fake',
                        help='type of running: freeze_fake, optimized_fake, perturbed_real')
    parser.add_argument('--purturb_radius', type=float, default=2.5,
                        help='2.5 for 8Gaussians, 1.5 for swissroll')

    parser.add_argument('--ratio', type=float, default=0.9,
                        help='2.5 for 8Gaussians, 1.5 for swissroll')

    parser.add_argument('--teacher_name', type=str, default='default', 
                        help='teacher options: default | scalized | mcts | rollout')
    parser.add_argument('--rollout_rate', type=float, default=5e-3, help='rollout rate')
    parser.add_argument('--rollout_method', type=str, default='ladam')
    parser.add_argument('--rollout_steps', type=int, default=50)
    parser.add_argument('--d_steps', type=int, default=1)
    parser.add_argument('--g_steps', type=int, default=1)
    parser.add_argument('--res', type=int, default=200, help='Resolution of Board')

    return parser.parse_args()

def build_teacher(args, gan, sess, data):
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

    return teacher

def create_prefix(args):
    prefix = 'figs/' + str(args.nlayers) + 'layer_' + str(args.scale) + '_gan/'

    prefix += args.mode + '_' + args.teacher_name + '_' + args.dataset + '_lrg_' + str(args.lrg) + '_lrd_' + str(args.lrd) + '_ds_' + str(args.d_steps) + '_' + args.refine_method + '/'
    if args.teacher_name == 'rollout':
        prefix += str(args.rollout_steps) + args.rollout_method + str(args.rollout_rate) + '/'
    prefix += time.strftime("%m-%d-%H-%M-%S") + '/'

    if os.path.exists(prefix):
        shutil.rmtree(prefix)
        print("remove folder " + prefix + " recursively")
    os.makedirs(prefix)

    return prefix

def train_optimal_gan(args):

    graph_opt_gan = tf.Graph()

    prefix = create_prefix(args)

    # open session
    with tf.Session(graph=graph_opt_gan) as sess:
        # declare instance for GAN
        noise = NoiseDataset()
        data = ToyDataset(distr=args.dataset, scale=args.scale, ratio=args.ratio)
        gan = G2N(sess,args,data,noise)
        if args.dataset == "8Gaussians":
            noise_sigma = 0.05
        if args.dataset == "Imbal-8Gaussians":
            noise_sigma = 0.05
        elif args.dataset == "swissroll":
            noise_sigma = 0.25
        else:
            None

        # build graph
        gan.build_model()
        teacher = build_teacher(args, gan, sess, data)

        # initialize all variables
        tf.global_variables_initializer().run()

        # restore model 
        if args.mode == "refinement":
            ckpt_file = args.dir_model + args.dataset + "-" + str(args.ckpt_num)
            gan.saver.restore(sess, ckpt_file)
            print(" [*] Success to read {}".format(ckpt_file))

        # graph inputs for visualize training results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 6))

        # settings
        norm_grad_list, g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list = [], [], [], [], []
        perturb_sample = None

        # plots
        noise_batch = noise.next_batch(args.batch_size)
        noise_sample = sess.run(gan.generates, feed_dict={gan.z: noise_batch})
        real_batch = data.next_batch(args.batch_size)  
        if args.dataset == "swissroll":
            region = args.scale * 1.3    
        else:
            region = args.scale * 0.8
        fname = args.dataset + "_" + args.mode + "_" + args.refine_method + "_ckpt_" + str(args.ckpt_num) + "_origin" + ".png"            
        draw_sample(noise_sample,real_batch,region,fname)
        fname = args.dataset + "_" + args.mode + "_" + args.refine_method + "_ckpt_" + str(args.ckpt_num) + "_real" + ".png"
        draw_sample(None,real_batch,region,fname)

        mean_dist, cnt_good = metrics_distance(real_batch,noise_sample,noise_sigma*3)
        print("Origin: mean_dist = %.2f, cnt_good = %.2f" % (mean_dist, cnt_good))
        
        # loop for epoch
        start_time = time.time()
        for it in range(args.niters):

            for i in range(args.d_steps):
                # create data
                real_batch = data.next_batch(args.batch_size)
                noise_batch = noise.next_batch(args.batch_size)

                if args.mode == "training":
                    # update D network from noise
                    _, d_loss, d_loss_fake, d_loss_real = gan.sess.run([gan.d_optim, gan.d_loss, gan.d_loss_fake, gan.d_loss_real],
                                        feed_dict={gan.inputs: real_batch, gan.z: noise_batch})
                elif args.mode == "refinement":
                    if args.refine_method == "freeze_fake":
                        noise_sample = sess.run(gan.generates, feed_dict={gan.z: noise_batch})
                        _, d_loss, d_loss_fake, d_loss_real = gan.sess.run([gan.d_optim, gan.d_loss, gan.d_loss_fake, gan.d_loss_real],
                                            feed_dict={gan.inputs: real_batch, gan.generates: noise_sample})
                    elif args.refine_method == "optimized_fake":
                        noise_sample, noise_grad, noise_sigmoid = sess.run([gan.generates, gan.grad_default, gan.fake_sigmoid], feed_dict={gan.z: noise_batch})
                        _, teacher_sample = teacher.manipulate_gradient(noise_sample, noise_sigmoid, noise_grad)
                        _, d_loss, d_loss_fake, d_loss_real = gan.sess.run([gan.d_optim, gan.d_loss, gan.d_loss_fake, gan.d_loss_real],
                                        feed_dict={gan.inputs: real_batch, gan.generates: teacher_sample})
                    elif args.refine_method == "perturbed_real":                
                        # r_perturb = torch.randn(args.batch_size,1) * 0.0 + args.purturb_radius
                        # th_perturb = torch.rand(args.batch_size,1) * 6.28
                        # perturb_sample = real_batch + torch.cat((r_perturb*torch.cos(th_perturb),r_perturb*torch.sin(th_perturb)),1)
                        perturb_noise = torch.from_numpy(random_n_sphere(args.batch_size,2)).type(torch.FloatTensor)
                        perturb_sample = real_batch + perturb_noise
                        _, d_loss, d_loss_fake, d_loss_real = gan.sess.run([gan.d_optim, gan.d_loss, gan.d_loss_fake, gan.d_loss_real],
                                            feed_dict={gan.inputs: real_batch, gan.generates: perturb_sample})
                    else: 
                        raise NotImplementedError 
                else:
                    raise NotImplementedError 

            d_loss_list.append((it,d_loss))
            d_loss_fake_list.append((it,d_loss_fake))
            d_loss_real_list.append((it,d_loss_real))

            # fetch G feedback (discriminative value + gradient) 

            for i in range(args.g_steps):
                noise_batch = noise.next_batch(args.batch_size)
                noise_sample, noise_grad, noise_sigmoid = sess.run([gan.generates, gan.grad_default, gan.fake_sigmoid], feed_dict={gan.z: noise_batch})
                
                if args.mode == "training":
                    if teacher:
                        teacher_grad, teacher_sample = teacher.manipulate_gradient(noise_sample, noise_sigmoid, noise_grad)
                        _, g_loss, summary_str = sess.run([gan.g_optim, gan.g_loss, gan.g_sum], feed_dict={gan.z: noise_batch, gan.grad_plugin: teacher_grad})
                    else:
                        teacher_grad, teacher_sample = None, None
                        _, g_loss, summary_str = sess.run([gan.g_optim, gan.g_loss, gan.g_sum], feed_dict={gan.z: noise_batch, gan.grad_plugin: noise_grad})
                elif args.mode == "refinement" and it % args.plot_every == 0:
                    teacher_grad, teacher_sample = teacher.manipulate_gradient(noise_sample, noise_sigmoid, noise_grad)
                    g_loss, summary_str = sess.run([gan.g_loss, gan.g_sum], feed_dict={gan.z: noise_batch})
                else:
                    None

            # g_loss_list.append((it,g_loss))

            # display training status
            if it % args.plot_every == 0:

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

                # compute metrics
                grid_grad_default, grid_sigmoid = sess.run([gan.fake_saliency, gan.fake_sigmoid], feed_dict={gan.fake_samples: grid_batch})
                if teacher:
                    grid_grad_teacher, _ = teacher.manipulate_gradient(grid_batch, grid_sigmoid, grid_grad_default)
                else:
                    grid_grad_teacher = None

                norm_grad = np.linalg.norm(grid_grad_default, axis=1)
                norm_median = np.median(norm_grad)
                norm_grad_list.append((it,norm_median))

                norm_noise_grad = np.linalg.norm(noise_grad, axis=1)
                norm_noise_grad_mean = np.mean(norm_noise_grad)

                # log
                if args.mode == "training":
                    mean_dist, cnt_good = metrics_distance(real_batch,noise_sample,noise_sigma*3)
                elif args.mode == "refinement":
                    mean_dist, cnt_good = metrics_distance(real_batch,teacher_sample,noise_sigma*3)
                print("Iter: %d, d_loss: %.4f, g_loss = %.4f, mean_dist = %.2f, cnt_good = %.2f" % (it, d_loss, g_loss, mean_dist, cnt_good))
                    
                # plots
                if args.mode == "training":
                    fname = args.dataset + "_" + args.mode + "_" + str(args.ratio) + "_" + str(it) + ".png"
                    draw_sample(noise_sample,real_batch,region,fname)
                elif args.mode == "refinement":
                    fname = args.dataset + "_" + args.mode + "_" + args.refine_method + "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + ".png"
                    draw_sample(teacher_sample,real_batch,region,fname)
                    draw_landscape(grid_batch,grid_sigmoid,real_batch,region,fname)

                # title = args.mode
                # if args.mode == "refinement":
                #     title += " | " + args.refine_method
                # plot_samples(prefix, ax1, args.scale, real_batch, None, None, None, it, g_loss, norm_noise_grad_mean, noise_sample, None, perturb_sample, xmedian - grid_length / 2.0, xmedian + grid_length / 2.0, ymedian - grid_length / 2.0, ymedian + grid_length / 2.0, None, None)
                # plot_samples(prefix, ax2, args.scale, real_batch, None, None, None, it, g_loss, norm_noise_grad_mean, None, teacher_sample, None, xmedian - grid_length / 2.0, xmedian + grid_length / 2.0, ymedian - grid_length / 2.0, ymedian + grid_length / 2.0, None, title)
                # plot_samples(prefix, ax3, args.scale, real_batch, grid_batch, grid_grad_default, grid_sigmoid, it, g_loss, None, None, None, None, xmedian - grid_length / 2.0, xmedian + grid_length / 2.0, ymedian - grid_length / 2.0, ymedian + grid_length / 2.0, grid_grad_teacher, None)   
                # plot_norm_distribution(ax4, 1 - noise_sigmoid, noise_grad, teacher_grad, args.teacher_name, args.rollout_method, args.rollout_steps, args.rollout_rate)
                # # plot_norm_histogram(ax2, 1 - noise_sigmoid, noise_grad, teacher_grad, args.teacher_name, args.rollout_method, args.rollout_steps, args.rollout_rate)
                # fig.savefig(prefix + 'fig_%05d.png' % it, bbox_inches='tight')
            
            # save training
            if args.save_model and it % 1000 == 0 and it > 0:
                gan.saver.save(sess, args.dir_model + args.dataset, global_step=it)

        # gan.saver.save(sess, args.dir_model + args.dataset)

        # plot_loss(prefix, norm_grad_list, g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list, args.teacher_name, args.rollout_method, args.rollout_steps, args.rollout_rate)
        # dump_loss(norm_grad_list, g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list, args)

if __name__ == '__main__':
    # print settings
    np.set_printoptions(precision=8, suppress=True)
    
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    else:
        print(args)

    # main functions
    train_optimal_gan(args)
