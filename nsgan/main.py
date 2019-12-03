import os
import argparse
import random
import numpy as np
import tensorflow as tf

from GAN import GAN
from utils import check_folder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

random.seed(2019)
np.random.seed(2019)
os.environ['PYTHONHASHSEED'] = str(2019)
tf.set_random_seed(2019)

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of collaborative sampling for NS-GAN and CGAN"
    parser = argparse.ArgumentParser(description=desc)
    # GAN
    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN', 'CGAN'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate for network update, 0.0002 for training')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--epoch', type=int, default=3, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--eval_size', type=int, default=50000, help='The size of batch')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    # stage
    parser.add_argument('--mode', type=str, default='train',
                        help='type of running: train, shape, calibrate, test')
    parser.add_argument('--method', type=str, default='benchmark',
                        help='type of running: standard, refinement, rejection, hastings, benchmark')
    # sampling
    parser.add_argument('--ckpt_num', type=int, default=200, help='ckpt to load')
    parser.add_argument('--rollout_steps', type=int, default=50, help='rollout steps for sample refinement')
    parser.add_argument('--rollout_rate', type=float, default=0.1, help='rollout rate for sample refinement')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        if args.gan_type == 'GAN':
            gan = GAN(sess,
                    epoch=args.epoch,
                    batch_size=args.batch_size,
                    eval_size=args.eval_size,
                    z_dim=args.z_dim,
                    dataset_name=args.dataset,
                    checkpoint_dir=args.checkpoint_dir,
                    result_dir=args.result_dir,
                    log_dir=args.log_dir)
        else:
            raise NotImplementedError

        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model(learning_rate=args.learning_rate, rollout_steps=args.rollout_steps, rollout_rate=args.rollout_rate)

        # launch the graph in a session
        if args.mode == "train":
            gan.train(mode=args.mode, method=args.method)
            print(" [*] Training finished!")
            gan.visualize_results(args.epoch-1)
            print(" [*] Testing finished!")
        elif args.mode == "shape" or args.mode == "calibrate":
            gan.shape(mode=args.mode, method=args.method, ckpt_num=args.ckpt_num)
            print(" [*] Shaping finished!")
        else:
            pass

if __name__ == '__main__':
    main()
