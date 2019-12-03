from __future__ import division
import os
import sys
import shutil
import random
import argparse
import matplotlib
import numpy as np
import tensorflow as tf
# additional path
sys.path.append(os.path.join('..', 'sampling'))
from GAN import GAN
from Datasets import *
from refiner_cpu import Refiner
from utils_sampling import draw_sample, draw_landscape, draw_kde, metrics_distance, metrics_diversity, metrics_distribution, calibration_diagnostic

matplotlib.use('Agg')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

random.seed(2019)
np.random.seed(2019)
os.environ['PYTHONHASHSEED'] = str(2019)

"""parsing and configuration"""
def parse_args():
    """ args for 2d synthetic """
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    # network
    parser.add_argument('--nhidden', type=int, default=64, help='number of hidden neurons')
    parser.add_argument('--nlayers', type=int, default=6, help='number of hidden layers')
    # training
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--lrg', type=float, default=5e-3, help='lr for G')
    parser.add_argument('--lrd', type=float, default=1e-2, help='lr for D')
    parser.add_argument('--d_steps', type=int, default=1)
    parser.add_argument('--g_steps', type=int, default=1)
    parser.add_argument('--niters', type=int, default=20001, help='number of iterations')
    # folder
    parser.add_argument('--save_model', action='store_true', help='save model flag')
    parser.add_argument('--dir_model', type=str, default='ckpt/',
                        help='Directory name to save model')
    # dataset
    parser.add_argument('--dataset', type=str, default='Imbal-8Gaussians',
                        help='dataset to use: 8Gaussians | 25Gaussians | swissroll | mnist')
    parser.add_argument('--scale', type=float, default=10., help='data scaling')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio of imbalance')
    # stage
    parser.add_argument('--mode', type=str, default='train',
                        help='type of running: train, shape, calibrate, test')
    parser.add_argument('--method', type=str, default='standard',
                        help='type of running: standard, refinement, rejection, hastings, benchmark')
    # sampling
    parser.add_argument('--ckpt_num', type=int, default=0, help='ckpt number')
    parser.add_argument('--rollout_rate', type=float, default=0.1, help='rollout rate')
    parser.add_argument('--rollout_method', type=str, default='ladam')
    parser.add_argument('--rollout_steps', type=int, default=50)
    # log
    parser.add_argument('--eval_every', type=int, default=1000, help='ckpt number')
    parser.add_argument('--eval_type', type=str, default='full', help='figs, logs, full')
    return parser.parse_args()

def build_refiner(args, gan, sess, data):
    """ collaborative sampling method """
    refiner = Refiner(args)
    refiner.set_env(gan, sess, data)
    return refiner

def build_rejector():
    """ rejection sampling method """
    from rejector import Rejector
    rejector = Rejector()
    return rejector

def build_independence_sampler():
    """ metropolis hasting method """
    from idpsampler import IndependenceSampler
    mh_idp_sampler = IndependenceSampler(T=20)
    return mh_idp_sampler

def create_folder(args):
    """ create output folder """
    prefix = 'figs/'
    prefix += args.mode + '_' + args.method + '_' + \
            str(args.ckpt_num) + '_' + args.dataset + '_' + \
            str(args.nlayers) + '_' + str(args.nhidden) + '_' + \
            str(args.lrd) + '_' + str(args.lrg) + '_' + \
            str(args.rollout_rate) + '/'

    if os.path.exists(prefix):
        shutil.rmtree(prefix)
        print("remove folder " + prefix + " recursively")
    os.makedirs(prefix)

    logdir = "log/" + args.dataset + '_' + args.mode + '/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(args.dir_model):
        os.makedirs(args.dir_model)

    return prefix, logdir

def save_metrics(filename, ckpt, iteration, mean_dist, cnt_good, kl_div, js_div, eff, z_calib, brier_score, ece, mce):
    file = open(filename, "a+")
    file.write("%d    %d    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f\r\n" \
        % (ckpt, iteration, mean_dist, cnt_good, kl_div, js_div, eff, z_calib, brier_score, ece, mce))
    file.close()

def evaluate(sess, gan, data, noise, args, target_batch, grid_batch, eval_batch, rejector, mh_idp_sampler, refiner, prefix, logdir, it, region):
    """ evaluate models and methods """
    # sigmoid
    grid_sigmoid = sess.run(gan.fake_sigmoid, feed_dict={gan.fake_samples: grid_batch})
    real_sigmoid = sess.run(gan.fake_sigmoid, feed_dict={gan.fake_samples: target_batch})
    eval_sigmoid = sess.run(gan.fake_sigmoid, feed_dict={gan.fake_samples: eval_batch})

    # discriminator properties
    figname = prefix + "reliability_" + args.dataset + "_" + args.mode + \
            "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + "_" + \
            str(args.rollout_rate) + "_" + str(args.rollout_steps) + "_" + str(args.ratio) + ".png"
    z_calib, brier_score, ece, mce = calibration_diagnostic(eval_sigmoid, real_sigmoid, figname)
    print("Iter %d: dawid = %.2f, brier = %.2f, ece = %.2f, mce = %.2f" % (it, z_calib, brier_score, ece, mce))

    # discriminator landscape
    fname = prefix + "critic_" + args.dataset + "_" + args.mode + \
            "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + "_" + \
            str(args.rollout_rate) + "_" + str(args.rollout_steps) + "_" + str(args.ratio) + ".png"
    draw_landscape(grid_batch, grid_sigmoid, target_batch, region, fname)

    # samples
    eval_size = eval_batch.shape[0]
    prefix_sample = prefix + "samples_"
    prefix_kde = prefix + "kde_"
    suffix = str(args.ratio) + "_" + str(it) + "_" + str(args.rollout_rate) + "_" + str(args.rollout_steps) + "_" + str(args.ratio) + ".png"
    if args.method == "standard" or args.method == "benchmark":
        if args.eval_type == "figs" or args.eval_type == "full":
            fname = prefix_sample + args.dataset + "_" + args.mode + "_standard_" + suffix
            draw_sample(eval_batch, target_batch, region, fname)
            fname = prefix_kde + args.dataset + "_" + args.mode + "_" + args.method + "_" + suffix
            draw_kde(eval_batch, region, fname)
        if args.eval_type == "logs" or args.eval_type == "full":
            mean_dist, cnt_good = metrics_distance(eval_batch, data.centeroids, data.std*4)
            kl_div = metrics_diversity(target_batch, eval_batch, data.centeroids, data.std*4)
            js_div = metrics_distribution(target_batch, eval_batch, data.centeroids, data.std*4)
            save_metrics(logdir+args.mode+"_standard.txt", args.ckpt_num, it, mean_dist, cnt_good, kl_div, js_div, 1.0, z_calib, brier_score, ece, mce)
            print("Iter (standard): %d, mean_dist = %.2f, cnt_good = %.2f, kl_div = %.2f, js_div = %.2f" % (it, mean_dist, cnt_good, kl_div, js_div))
    if (args.method == "rejection" or args.method == "benchmark") and args.mode == "calibrate":
        cnt_propose = eval_size
        accepted_sample = np.empty_like(eval_batch)
        rejector.set_score_max(np.amax(real_sigmoid))
        accepted_base = rejector.sampling(eval_batch, eval_sigmoid, shift_percent=100.0)
        cnt_reject = accepted_base.shape[0]
        if cnt_reject > 0:
            accepted_sample[:cnt_reject] = accepted_base
        while cnt_reject < eval_size:
            batch_noise = noise.next_batch(eval_size)
            batch_extra = sess.run(gan.generates, feed_dict={gan.z: batch_noise})
            batch_sigmoid = sess.run(gan.fake_sigmoid, feed_dict={gan.fake_samples: batch_extra})
            accepted_extra = rejector.sampling(batch_extra, batch_sigmoid, shift_percent=100.0)
            cnt_extra = accepted_extra.shape[0]
            if cnt_extra > 0:
                if cnt_reject + cnt_extra < eval_size:
                    accepted_sample[cnt_reject:cnt_reject+cnt_extra] = accepted_extra
                else:
                    accepted_sample[cnt_reject:eval_size] = accepted_extra[:eval_size-cnt_reject]
                cnt_reject = cnt_reject + cnt_extra
                cnt_propose = cnt_propose + eval_size
        if args.eval_type == "figs" or args.eval_type == "full":
            fname = prefix_sample + args.dataset + "_rejection_" + args.method + "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + "_" + str(args.ratio) + ".png"
            draw_sample(accepted_sample, target_batch, region, fname)
            fname = prefix_kde + args.dataset + "_rejection_" + args.method + "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + "_" + str(args.ratio) + ".png"
            draw_kde(accepted_sample, region, fname)
        if args.eval_type == "logs" or args.eval_type == "full":
            mean_dist, cnt_good = metrics_distance(accepted_sample, data.centeroids, data.std*4)
            kl_div = metrics_diversity(target_batch, accepted_sample, data.centeroids, data.std*4)
            js_div = metrics_distribution(target_batch, accepted_sample, data.centeroids, data.std*4)
            eff = cnt_reject / cnt_propose
            print("Iter (rejection): %d, mean_dist = %.2f, cnt_good = %.2f, kl_div = %.2f, js_div = %.2f, eff = %.2f" % (it, mean_dist, cnt_good, kl_div, js_div, eff))
            save_metrics(logdir+args.mode+"_rejection.txt", args.ckpt_num, it, mean_dist, cnt_good, kl_div, js_div, eff, z_calib, brier_score, ece, mce)
    if (args.method == "hastings" or args.method == "benchmark") and args.mode == "calibrate":
        cnt_propose = eval_size
        accepted_sample = np.empty_like(eval_batch)
        mh_idp_sampler.set_score_curr(np.mean(real_sigmoid))
        accepted_base = mh_idp_sampler.sampling(eval_batch, eval_sigmoid)
        cnt_reject = accepted_base.shape[0]
        if cnt_reject > 0:
            accepted_sample[:cnt_reject] = accepted_base
        while cnt_reject < eval_size:
            batch_noise = noise.next_batch(eval_size)
            batch_extra = sess.run(gan.generates, feed_dict={gan.z: batch_noise})
            batch_sigmoid = sess.run(gan.fake_sigmoid, feed_dict={gan.fake_samples: batch_extra})
            accepted_extra = mh_idp_sampler.sampling(batch_extra, batch_sigmoid)
            cnt_extra = accepted_extra.shape[0]
            if cnt_extra > 0:
                if cnt_reject + cnt_extra < eval_size:
                    accepted_sample[cnt_reject:cnt_reject+cnt_extra] = accepted_extra
                else:
                    accepted_sample[cnt_reject:eval_size] = accepted_extra[:eval_size-cnt_reject]
                cnt_reject = cnt_reject + cnt_extra
                cnt_propose = cnt_propose + eval_size
        if args.eval_type == "figs" or args.eval_type == "full":
            fname = prefix_sample + args.dataset + "_hastings_" + args.method + "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + "_" + str(args.ratio) + ".png"
            draw_sample(accepted_sample, target_batch, region, fname)
            fname = prefix_kde + args.dataset + "_hastings_" + args.method + "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + "_" + str(args.ratio) + ".png"
            draw_kde(accepted_sample, region, fname)
        if args.eval_type == "logs" or args.eval_type == "full":
            mean_dist, cnt_good = metrics_distance(accepted_sample, data.centeroids, data.std*4)
            kl_div = metrics_diversity(target_batch, accepted_sample, data.centeroids, data.std*4)
            js_div = metrics_distribution(target_batch, accepted_sample, data.centeroids, data.std*4)
            eff = cnt_reject / cnt_propose
            print("Iter (hastings): %d, mean_dist = %.2f, cnt_good = %.2f, kl_div = %.2f, js_div = %.2f, eff = %.2f" % (it, mean_dist, cnt_good, kl_div, js_div, eff))
            save_metrics(logdir+args.mode+"_hastings.txt", args.ckpt_num, it, mean_dist, cnt_good, kl_div, js_div, eff, z_calib, brier_score, ece, mce)
    if (args.method == "refinement" or args.method == "benchmark") and args.mode == "shape":
        # refinement step
        refined_sample = refiner.manipulate_sample(eval_batch)
        if args.eval_type == "figs" or args.eval_type == "full":
            fname = prefix_sample + args.dataset + "_refinement_" + args.method + "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + "_" + str(args.rollout_rate) + "_" + str(args.rollout_steps) + "_" + str(args.ratio) + ".png"
            draw_sample(refined_sample, target_batch, region, fname)
            fname = prefix_kde + args.dataset + "_refinement_" + args.method + "_" + suffix
            draw_kde(refined_sample, region, fname)
        if args.eval_type == "logs" or args.eval_type == "full":
            mean_dist, cnt_good = metrics_distance(refined_sample, data.centeroids, data.std*4)
            kl_div = metrics_diversity(target_batch, refined_sample, data.centeroids, data.std*4)
            js_div = metrics_distribution(target_batch, refined_sample, data.centeroids, data.std*4)
            print("Iter (refinement): %d, mean_dist = %.2f, cnt_good = %.2f, kl_div = %.2f, js_div = %.2f" % (it, mean_dist, cnt_good, kl_div, js_div))
            save_metrics(logdir+args.mode+"_refinement.txt", args.ckpt_num, it, mean_dist, cnt_good, kl_div, js_div, 1.0, z_calib, brier_score, ece, mce)
        # rejection step
        cnt_propose = eval_size
        accepted_sample = np.empty_like(eval_batch)
        refined_sigmoid = sess.run(gan.fake_sigmoid, feed_dict={gan.fake_samples: refined_sample})
        mh_idp_sampler.set_score_curr(np.mean(real_sigmoid))
        accepted_base = mh_idp_sampler.sampling(refined_sample, refined_sigmoid)
        cnt_reject = accepted_base.shape[0]
        if cnt_reject > 0:
            accepted_sample[:cnt_reject] = accepted_base
        while cnt_reject < eval_size:
            batch_noise = noise.next_batch(eval_size)
            batch_extra = sess.run(gan.generates, feed_dict={gan.z: batch_noise})
            refined_extra = refiner.manipulate_sample(batch_extra)
            refined_sigmoid = sess.run(gan.fake_sigmoid, feed_dict={gan.fake_samples: refined_extra})
            accepted_extra = mh_idp_sampler.sampling(refined_extra, refined_sigmoid)
            cnt_extra = accepted_extra.shape[0]
            if cnt_extra > 0:
                if cnt_reject + cnt_extra < eval_size:
                    accepted_sample[cnt_reject:cnt_reject+cnt_extra] = accepted_extra
                else:
                    accepted_sample[cnt_reject:eval_size] = accepted_extra[:eval_size-cnt_reject]
                cnt_reject = cnt_reject + cnt_extra
                cnt_propose = cnt_propose + eval_size
        if args.eval_type == "figs" or args.eval_type == "full":
            fname = prefix_sample + args.dataset + "_refinement_rejection_" + args.method + "_ckpt_" + str(args.ckpt_num) + "_" + str(it) + "_" + str(args.rollout_rate) + "_" + str(args.rollout_steps) + "_" + str(args.ratio) + ".png"
            draw_sample(accepted_sample, target_batch, region, fname)
            fname = prefix_kde + args.dataset + "_refinement_rejection" + args.method + "_" + suffix
            draw_kde(accepted_sample, region, fname)
        if args.eval_type == "logs" or args.eval_type == "full":
            mean_dist, cnt_good = metrics_distance(accepted_sample, data.centeroids, data.std*4)
            kl_div = metrics_diversity(target_batch, accepted_sample, data.centeroids, data.std*4)
            js_div = metrics_distribution(target_batch, accepted_sample, data.centeroids, data.std*4)
            eff = cnt_reject / cnt_propose
            print("Iter (collaborate): %d, mean_dist = %.2f, cnt_good = %.2f, kl_div = %.2f, js_div = %.2f, eff = %.2f" % (it, mean_dist, cnt_good, kl_div, js_div, eff))
            save_metrics(logdir+args.mode+"_collaborate.txt", args.ckpt_num, it, mean_dist, cnt_good, kl_div, js_div, eff, z_calib, brier_score, ece, mce)

def main(args):
    """ main function - train models and apply sampling methods """

    prefix, logdir = create_folder(args)

    tf.reset_default_graph()
    tf.set_random_seed(2019)

    # open session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # declare instance for GAN
        noise = NoiseDataset()
        data = ToyDataset(distr=args.dataset, scale=args.scale, ratio=args.ratio)
        gan = GAN(sess, args, data, noise)

        # build graph
        gan.build_model()
        refiner = build_refiner(args, gan, sess, data)
        rejector = build_rejector()
        mh_idp_sampler = build_independence_sampler()

        # initialize all variables
        sess.run(tf.global_variables_initializer())

        # restore model
        if not args.mode == "train":
            ckpt_file = args.dir_model + args.dataset + "-" + str(args.ckpt_num)
            gan.saver.restore(sess, ckpt_file)
            print(" [*] Success to read {}".format(ckpt_file))

        # visualization
        eval_size = 10000
        eval_noise = noise.next_batch(eval_size)
        eval_batch = sess.run(gan.generates, feed_dict={gan.z: eval_noise})

        target_batch = data.next_batch(eval_size)

        if args.dataset == "25Gaussians":
            region = args.scale * 2.4
        elif args.dataset == "swissroll":
            region = args.scale * 1.3
        else:
            region = args.scale * 0.8
        fname = prefix + args.dataset + "_" + args.mode + "_" + args.method + "_ckpt_" + str(args.ckpt_num) + "_origin" + ".png"
        draw_sample(eval_batch, target_batch, region, fname)
        fname = prefix + args.dataset + "_" + args.mode + "_" + args.method + "_ckpt_" + str(args.ckpt_num) + "_real" + ".png"
        draw_sample(None, target_batch, region, fname)
        fname = prefix + "eval_" + str(args.ckpt_num) + "_kde" + ".png"
        draw_kde(eval_batch, region, fname)

        ngrids = 21
        grid_batch = np.zeros((ngrids * ngrids, 2))
        if args.dataset == "25Gaussians":
            grid_region = args.scale * 2.5
        else:
            grid_region = args.scale

        step = 2 * grid_region / (ngrids-1)
        for i in range(ngrids):
            for j in range(ngrids):
                grid_batch[i * ngrids + j, 0] = -grid_region + i * step
                grid_batch[i * ngrids + j, 1] = -grid_region + j * step

        grid_sigmoid = sess.run(gan.fake_sigmoid, feed_dict={gan.fake_samples: grid_batch})
        fname = prefix + "critic_" + args.dataset + "_" + args.mode + "_" + args.method + "_ckpt_" + str(args.ckpt_num) + "_landscape" + ".png"
        draw_landscape(grid_batch, grid_sigmoid, target_batch, region, fname)

        # metrics
        fname = prefix + "real_kde.png"
        draw_kde(target_batch, region, fname)

        mean_dist, cnt_good = metrics_distance(target_batch, data.centeroids, data.std*4)
        kl_real = metrics_diversity(target_batch, target_batch, data.centeroids, data.std*4)
        js_real = metrics_distribution(target_batch, target_batch, data.centeroids, data.std*4)
        print("Metrics: (real) mean_dist = %.2f, cnt_good = %.2f, kl = %.2f, js = %.2f" % (mean_dist, cnt_good, kl_real, js_real))

        mean_dist, cnt_good = metrics_distance(eval_batch, data.centeroids, data.std*4)
        kl_sample = metrics_diversity(target_batch, eval_batch, data.centeroids, data.std*4)
        js_sample = metrics_distribution(target_batch, eval_batch, data.centeroids, data.std*4)
        print("Metrics: (model) mean_dist = %.2f, cnt_good = %.2f, kl = %.2f, js = %.2f" % (mean_dist, cnt_good, kl_sample, js_sample))

        # loop for epoch
        for i in range(args.niters):

            # update discriminator
            for _ in range(args.d_steps):

                # create data
                real_batch = data.next_batch(args.batch_size)
                noise_batch = noise.next_batch(args.batch_size)

                if args.mode == "train":
                    # train D network
                    _ = gan.sess.run(gan.d_optim, feed_dict={gan.inputs: real_batch, gan.z: noise_batch})
                elif args.mode == "calibrate":
                    # calibrate D network
                    noise_sample = sess.run(gan.generates, feed_dict={gan.z: noise_batch})
                    _ = gan.sess.run(gan.d_optim, feed_dict={gan.inputs: real_batch, gan.generates: noise_sample})
                elif args.mode == "shape":
                    # shape D network
                    noise_sample, noise_grad = sess.run([gan.generates, gan.grad_default], feed_dict={gan.z: noise_batch})
                    refined_sample = refiner.manipulate_sample(noise_sample, 'probabilistic')
                    _ = gan.sess.run(gan.d_optim, feed_dict={gan.inputs: real_batch, gan.generates: refined_sample})
                elif args.mode == "test":
                    pass
                else:
                    raise NotImplementedError

            # update generator
            for _ in range(args.g_steps):
                if args.mode == "train":
                    noise_batch = noise.next_batch(args.batch_size)
                    noise_sample, noise_grad = sess.run([gan.generates, gan.grad_default], feed_dict={gan.z: noise_batch})
                    _ = sess.run(gan.g_optim, feed_dict={gan.z: noise_batch, gan.grad_plugin: noise_grad})
                else:
                    pass

            # evaluation
            if i % args.eval_every == 0:
                # update valuation batch
                if args.mode == "train":
                    eval_batch = sess.run(gan.generates, feed_dict={gan.z: eval_noise})
                # run evaluation
                evaluate(sess, gan, data, noise, args, target_batch, grid_batch, eval_batch, rejector, mh_idp_sampler, refiner, prefix, logdir, i, region)

            # save training
            if args.save_model and i % args.eval_every == 0 and i > 0:
                gan.saver.save(sess, args.dir_model + args.dataset, global_step=i)

if __name__ == '__main__':
    # print settings
    np.set_printoptions(precision=8, suppress=True)

    # main functions
    main(parse_args())
