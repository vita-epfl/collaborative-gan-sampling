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

from G2N import G2N
from Datasets import * 

sys.path.append('.')
sys.path.append(os.path.join('..', 'sampling'))    
from refiner_cpu import Refiner
from utils import draw_sample, draw_landscape, draw_density, metrics_distance, metrics_diversity

np.random.seed(20190314)
tf.set_random_seed(20190314)

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    # network
    parser.add_argument('--nhidden', type=int, default=64, help='number of hidden neurons')
    parser.add_argument('--nlayers', type=int, default=6, help='number of hidden layers')
    # training 
    parser.add_argument('--batch_size', type=int, default=400, help='batch size')
    parser.add_argument('--lrg', type=float, default=5e-3, help='lr for G')
    parser.add_argument('--lrd', type=float, default=1e-2, help='lr for D')
    parser.add_argument('--d_steps', type=int, default=1)
    parser.add_argument('--g_steps', type=int, default=1)    
    parser.add_argument('--minibatch_discriminate', action='store_true', help='minibatch_discriminate flag')        
    parser.add_argument('--niters', type=int, default=5001, help='number of iterations')    
    # folder 
    parser.add_argument('--save_model', action='store_true', help='save model flag')
    parser.add_argument('--dir_model', type=str, default='ckpt/', help='Directory name to save training model')
    # dataset
    parser.add_argument('--dataset', type=str, default='Imbal-8Gaussians',
                        help='dataset to use: 8Gaussians | 25Gaussians | swissroll | mnist')
    parser.add_argument('--scale', type=float, default=10., help='data scaling')
    parser.add_argument('--ratio', type=float, default=0.9, help='ratio of imbalance')
    # stage
    parser.add_argument('--mode', type=str, default='training',
                        help='type of running: training, refinement, testing, rejection')
    # shaping
    parser.add_argument('--shaping_method', type=str, default='optimized_fake',
                        help='type of running: optimized_fake, freeze_fake, perturbed_real')
    # sampling
    parser.add_argument('--ckpt_num', type=int, default=2000, help='ckpt number')
    parser.add_argument('--rollout_rate', type=float, default=5e-3, help='rollout rate')
    parser.add_argument('--rollout_method', type=str, default='ladam')
    parser.add_argument('--rollout_steps', type=int, default=50)
    parser.add_argument('--rollout_layer', type=str, default='x', help='layer options: z | x')
    # log
    parser.add_argument('--plot_every', type=int, default=100, help='ckpt number')
    return parser.parse_args()

def build_refiner(args, gan, sess, data):
    # build refiner
    refiner = Refiner(args)
    refiner.set_env(gan, sess, data)
    return refiner

def build_rejector(args):
    from rejector import Rejector
    rejector = Rejector(args)    
    return rejector

def create_prefix(args):
    prefix = 'figs/' + str(args.nlayers) + 'layer_' + str(args.scale) + '_gan/'

    prefix += args.mode + '_' + args.dataset + '_lrg_' + str(args.lrg) + '_lrd_' + str(args.lrd) + '_ds_' + str(args.d_steps) + '_' + args.shaping_method + '/'
    prefix += str(args.rollout_steps) + args.rollout_method + str(args.rollout_rate) + '/'
    # prefix += time.strftime("%m-%d-%H-%M-%S") + '/'

    if os.path.exists(prefix):
        shutil.rmtree(prefix)
        print("remove folder " + prefix + " recursively")
    os.makedirs(prefix)

    if not os.path.exists(args.dir_model):
        os.makedirs(args.dir_model) 

    return prefix

def train_optimal_gan(args):

    prefix = create_prefix(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # open session
    with tf.Session(config=config) as sess:
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
        refiner = build_refiner(args, gan, sess, data)
        rejector = build_rejector(args)

        # initialize all variables
        tf.global_variables_initializer().run()

        # restore model 
        if not args.mode == "training":
            ckpt_file = args.dir_model + args.dataset + "-" + str(args.ckpt_num)
            gan.saver.restore(sess, ckpt_file)
            print(" [*] Success to read {}".format(ckpt_file))

        # graph inputs for visualize training results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 6))

        # settings
        norm_grad_list, g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list = [], [], [], [], []
        perturb_sample = None

        # plots
        eval_size = 10000
        noise_batch = noise.next_batch(args.batch_size)
        noise_sample = sess.run(gan.generates, feed_dict={gan.z: noise_batch})
        real_batch = data.next_batch(args.batch_size)  
        if args.dataset == "swissroll":
            region = args.scale * 1.3    
        else:
            region = args.scale * 0.8
        fname = prefix + args.dataset + "_" + args.mode + "_" + args.shaping_method + "_ckpt_" + str(args.ckpt_num) + "_origin" + ".png"            
        draw_sample(noise_sample,real_batch,region,fname)
        fname = prefix + args.dataset + "_" + args.mode + "_" + args.shaping_method + "_ckpt_" + str(args.ckpt_num) + "_real" + ".png"
        draw_sample(None,real_batch,region,fname)

        # grid visualization
        ngrids = 21
        grid_batch = np.zeros((ngrids * ngrids, 2))
        # grid_length = max(args.scale/2,max(np.ceil(np.max(fake_batch[:, 0])) - np.floor(np.min(fake_batch[:, 0])), np.ceil(np.max(fake_batch[:, 1])) - np.floor(np.min(fake_batch[:, 1]))))
        step = 2 * args.scale / (ngrids-1)
        for i in range(ngrids):
            for j in range(ngrids):
                grid_batch[i * ngrids + j, 0] = -args.scale + i * step
                grid_batch[i * ngrids + j, 1] = -args.scale + j * step

        # compute metrics
        _, grid_sigmoid = sess.run([gan.fake_saliency, gan.fake_sigmoid], feed_dict={gan.fake_samples: grid_batch})
        fname = prefix + "critic_" + args.dataset + "_" + args.mode + "_" + args.shaping_method + "_ckpt_" + str(args.ckpt_num) + "_landscape" + ".png"
        draw_landscape(grid_batch,grid_sigmoid,real_batch,region,fname)

        # fname = prefix + "real_landscape" + ".png"
        # real_batch = data.next_batch(10000)  
        # draw_density(real_batch.numpy(),region,fname)

        mean_dist, cnt_good = metrics_distance(real_batch.numpy(), data.centeroids, data.std*3)
        kl_real = metrics_diversity(real_batch.numpy(), real_batch.numpy(), data.centeroids, data.std*3)
        print("Metrics: (real) mean_dist = %.2f, cnt_good = %.2f, kl = %.2f" % (mean_dist, cnt_good, kl_real))

        mean_dist, cnt_good = metrics_distance(noise_sample, data.centeroids, data.std*3)
        kl_sample = metrics_diversity(real_batch.numpy(), noise_sample, data.centeroids, data.std*3)
        print("Metrics: (model) mean_dist = %.2f, cnt_good = %.2f, kl = %.2f" % (mean_dist, cnt_good, kl_sample))
        
        buf_mean_dist_refine, buf_cnt_good_refine, buf_kl_refine = [], [], []
        buf_mean_dist_reject, buf_cnt_good_reject, buf_kl_reject = [], [], []
        buf_mean_dist_both, buf_cnt_good_both, buf_kl_both = [], [], []

        real_batch = data.next_batch(10000)
        real_sigmoid = sess.run(gan.fake_sigmoid, feed_dict={gan.fake_samples: real_batch})
        rejector.set_score_max(np.amax(real_sigmoid))

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
                    if args.shaping_method == "freeze_fake":
                        noise_sample = sess.run(gan.generates, feed_dict={gan.z: noise_batch})
                        _, d_loss, d_loss_fake, d_loss_real = gan.sess.run([gan.d_optim, gan.d_loss, gan.d_loss_fake, gan.d_loss_real],
                                            feed_dict={gan.inputs: real_batch, gan.generates: noise_sample})
                    elif args.shaping_method == "optimized_fake":
                        noise_sample, noise_grad, noise_sigmoid = sess.run([gan.generates, gan.grad_default, gan.fake_sigmoid], feed_dict={gan.z: noise_batch})
                        _, refiner_sample = refiner.manipulate_gradient(noise_sample, noise_sigmoid, noise_grad)
                        _, d_loss, d_loss_fake, d_loss_real = gan.sess.run([gan.d_optim, gan.d_loss, gan.d_loss_fake, gan.d_loss_real],
                                        feed_dict={gan.inputs: real_batch, gan.generates: refiner_sample})
                    elif args.shaping_method == "perturbed_real":                
                        perturb_noise = torch.from_numpy(random_n_sphere(args.batch_size,2)).type(torch.FloatTensor)
                        perturb_sample = real_batch + perturb_noise
                        _, d_loss, d_loss_fake, d_loss_real = gan.sess.run([gan.d_optim, gan.d_loss, gan.d_loss_fake, gan.d_loss_real],
                                            feed_dict={gan.inputs: real_batch, gan.generates: perturb_sample})
                    else: 
                        raise NotImplementedError
                elif args.mode == "rejection":
                    pass 
                else:
                    raise NotImplementedError 

            # d_loss_list.append((it,d_loss))
            # d_loss_fake_list.append((it,d_loss_fake))
            # d_loss_real_list.append((it,d_loss_real))

            # fetch G feedback (discriminative value + gradient) 

            for i in range(args.g_steps):
                noise_batch = noise.next_batch(args.batch_size)
                noise_sample, noise_grad, z_grad, noise_sigmoid = sess.run([gan.generates, gan.grad_default, gan.grad_z, gan.fake_sigmoid], feed_dict={gan.z: noise_batch})

                if args.mode == "training":
                    # TODO: collaborative in training 
                    # if refiner:
                    #     refiner_grad, refiner_sample = refiner.manipulate_gradient(noise_sample, noise_sigmoid, noise_grad)
                    #     _, g_loss, summary_str = sess.run([gan.g_optim, gan.g_loss, gan.g_sum], feed_dict={gan.z: noise_batch, gan.grad_plugin: refiner_grad})
                    # else:
                    refiner_grad, refiner_sample = None, None
                    _, g_loss, summary_str = sess.run([gan.g_optim, gan.g_loss, gan.g_sum], feed_dict={gan.z: noise_batch, gan.grad_plugin: noise_grad})
                elif args.mode == "refinement" and it % args.plot_every == 0:
                    if args.rollout_layer == "x":
                        _, refiner_sample = refiner.manipulate_gradient(noise_sample, noise_sigmoid, noise_grad)
                    elif args.rollout_layer == "z":
                        refiner_sample = refiner.manipute_latent(noise_batch.numpy(), noise_sigmoid, np.array(z_grad))
                    else:
                        raise NotImplementedError 
                    g_loss, summary_str = sess.run([gan.g_loss, gan.g_sum], feed_dict={gan.z: noise_batch})
                elif args.mode == "rejection":
                    noise_sample, noise_sigmoid = sess.run([gan.generates, gan.fake_sigmoid], feed_dict={gan.z: noise_batch})
                    refiner_sample = rejector.sampling(noise_sample, noise_sigmoid)
                else:
                    None

            # g_loss_list.append((it,g_loss))

            # display training status
            if it % args.plot_every == 0:

                # display training status
                fake_batch, fake_grad = sess.run([gan.fake_samples, gan.fake_saliency], feed_dict={gan.z: noise_batch})

                # compute metrics
                grid_grad_default, grid_sigmoid = sess.run([gan.fake_saliency, gan.fake_sigmoid], feed_dict={gan.fake_samples: grid_batch})
                # if refiner:
                #     grid_grad_refiner, _ = refiner.manipulate_gradient(grid_batch, grid_sigmoid, grid_grad_default)
                # else:
                    # grid_grad_refiner = None

                norm_grad = np.linalg.norm(grid_grad_default, axis=1)
                norm_median = np.median(norm_grad)
                norm_grad_list.append((it,norm_median))

                norm_noise_grad = np.linalg.norm(noise_grad, axis=1)
                norm_noise_grad_mean = np.mean(norm_noise_grad)

                # plots
                if args.mode == "training":
                    fname = prefix + "samples_" + args.dataset + "_" + args.mode + "_" + str(args.ratio) + "_" + str(it) + "_" + str(args.rollout_rate) + "_" + str(args.rollout_steps) + "_" + str(args.ratio) + ".png"
                    draw_sample(noise_sample,real_batch,region,fname)
                elif args.mode == "refinement":
                    fname = prefix + "samples_" + args.dataset + "_" + args.mode + "_" + args.shaping_method + "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + "_" + str(args.rollout_rate) + "_" + str(args.rollout_steps) + "_" + str(args.ratio) + ".png"
                    draw_sample(refiner_sample,real_batch,region,fname)
                    fname = prefix + "critic_" + args.dataset + "_" + args.mode + "_" + args.shaping_method + "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + "_" + str(args.rollout_rate) + "_" + str(args.rollout_steps) + "_" + str(args.ratio) + ".png"                    
                    draw_landscape(grid_batch,grid_sigmoid,real_batch,region,fname)
                elif args.mode == "rejection":
                    fname = prefix + "samples_" + args.dataset + "_" + args.mode + "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + "_" + str(args.ratio) + ".png"
                    draw_sample(refiner_sample,real_batch,region,fname)                    

                # log
                noise_batch = noise.next_batch(eval_size)
                real_batch = data.next_batch(eval_size)
                noise_sample, noise_grad, z_grad, noise_sigmoid = sess.run([gan.generates, gan.grad_default, gan.grad_z, gan.fake_sigmoid], feed_dict={gan.z: noise_batch})

                if args.mode == "training":
                    mean_dist, cnt_good = metrics_distance(noise_sample, data.centeroids, data.std*3)
                    kl = metrics_diversity(real_batch.numpy(), noise_sample, data.centeroids, data.std*3)    
                    print("Iter: %d, d_loss: %.2f, g_loss = %.2f, mean_dist = %.2f, cnt_good = %.2f, kl = %.2f" % (it, d_loss, g_loss, mean_dist, cnt_good, kl))
                
                elif args.mode == "refinement":
                    if args.rollout_layer == "x":
                        _, refined_sample = refiner.manipulate_gradient(noise_sample, noise_sigmoid, noise_grad)
                    elif args.rollout_layer == "z":
                        refined_sample = refiner.manipute_latent(noise_batch.numpy(), noise_sigmoid, np.array(z_grad))
                    else:
                        raise NotImplementedError 
                    mean_dist, cnt_good = metrics_distance(refined_sample, data.centeroids, data.std*3)
                    kl = metrics_diversity(real_batch.numpy(), refined_sample, data.centeroids, data.std*3)
                    buf_mean_dist_refine.append(mean_dist)
                    buf_cnt_good_refine.append(cnt_good)
                    buf_kl_refine.append(kl)

                    refined_sigmoid = sess.run(gan.fake_sigmoid, feed_dict={gan.fake_samples: refined_sample})
                    accept_sample = rejector.sampling(refined_sample, refined_sigmoid)
                    mean_dist_accept, cnt_good_accept = metrics_distance(accept_sample, data.centeroids, data.std*3)
                    kl_accept = metrics_diversity(real_batch.numpy(), accept_sample, data.centeroids, data.std*3)
                    buf_mean_dist_both.append(mean_dist_accept)
                    buf_cnt_good_both.append(cnt_good_accept)
                    buf_kl_both.append(kl_accept)

                    print("Iter: %d, d_loss: %.2f, g_loss = %.2f, mean_dist = %.2f, cnt_good = %.2f, kl = %.2f, mean_dist_accept = %.2f, cnt_good_accept = %.2f, kl_accept = %.2f" % (it, d_loss, g_loss, mean_dist, cnt_good, kl, mean_dist_accept, cnt_good_accept, kl_accept))

                elif args.mode == "rejection":
                    accept_sample = rejector.sampling(noise_sample, noise_sigmoid)
                    mean_dist, cnt_good = metrics_distance(accept_sample, data.centeroids, data.std*3)
                    kl = metrics_diversity(real_batch.numpy(), accept_sample, data.centeroids, data.std*3)
                    buf_mean_dist_reject.append(mean_dist)
                    buf_cnt_good_reject.append(cnt_good)
                    buf_kl_reject.append(kl)

                else:
                    pass

                # title = args.mode
                # if args.mode == "refinement":
                #     title += " | " + args.shaping_method
                # plot_samples(prefix, ax1, args.scale, real_batch, None, None, None, it, g_loss, norm_noise_grad_mean, noise_sample, None, perturb_sample, xmedian - grid_length / 2.0, xmedian + grid_length / 2.0, ymedian - grid_length / 2.0, ymedian + grid_length / 2.0, None, None)
                # plot_samples(prefix, ax2, args.scale, real_batch, None, None, None, it, g_loss, norm_noise_grad_mean, None, refiner_sample, None, xmedian - grid_length / 2.0, xmedian + grid_length / 2.0, ymedian - grid_length / 2.0, ymedian + grid_length / 2.0, None, title)
                # plot_samples(prefix, ax3, args.scale, real_batch, grid_batch, grid_grad_default, grid_sigmoid, it, g_loss, None, None, None, None, xmedian - grid_length / 2.0, xmedian + grid_length / 2.0, ymedian - grid_length / 2.0, ymedian + grid_length / 2.0, grid_grad_refiner, None)   
                # plot_norm_distribution(ax4, 1 - noise_sigmoid, noise_grad, refiner_grad, args.refiner_name, args.rollout_method, args.rollout_steps, args.rollout_rate)
                # # plot_norm_histogram(ax2, 1 - noise_sigmoid, noise_grad, refiner_grad, args.refiner_name, args.rollout_method, args.rollout_steps, args.rollout_rate)
                # fig.savefig(prefix + 'fig_%05d.png' % it, bbox_inches='tight')
            
            # save training
            if args.save_model and it % 1000 == 0 and it > 0:
                gan.saver.save(sess, args.dir_model + args.dataset, global_step=it)

        if args.mode == "refinement":
            print("Metrics: (refine) mean_dist = %.2f, cnt_good = %.2f, kl = %.2f" % (np.min(buf_mean_dist_refine), np.max(buf_cnt_good_refine), np.min(buf_kl_refine) ))
            print("Metrics: (both) mean_dist = %.2f, cnt_good = %.2f, kl = %.2f" % (np.min(buf_mean_dist_both), np.max(buf_cnt_good_both), np.min(buf_kl_both) ))
        elif args.mode == "rejection":
            print("Metrics: (reject) mean_dist = %.2f, cnt_good = %.2f, kl = %.2f" % (np.min(buf_mean_dist_reject), np.max(buf_cnt_good_reject), np.min(buf_kl_reject) ))
        else:
            pass

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
