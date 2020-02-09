import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "Compare various sampling methods for GANs"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mode', type=str, default='benchmark',
                        help='type of running: train, shape, calibrate, benchmark')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='directory of logs for plots')
    return parser.parse_args()


def plot_method(x, y_r, y_g, y_b, y_c, y_k, label_r, label_g, label_b, label_c, label_k, label_x, label_y, xlim=None, ylim=None, fname=None):
    fig = plt.figure(figsize=(4, 3))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    ax.plot(x, y_r, 'r-x', label=label_r)
    ax.plot(x, y_g, 'y-o', label=label_g)
    ax.plot(x, y_b, 'b-*', label=label_b)
    ax.plot(x, y_c, 'c-^', label=label_c)
    ax.plot(x, y_k, 'k-s', label=label_k)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax.grid()
    plt.legend()
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
        print(fname)
    plt.close()


def plot_mode(x, y_r, y_g, y_b, label_r, label_g, label_b, label_x, label_y, ylim=None, fname=None):
    fig = plt.figure(figsize=(4, 3))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    ax.plot(x, y_r, 'r-x', label=label_r)
    ax.plot(x, y_g, 'g-o', label=label_g)
    ax.plot(x, y_b, 'b-*', label=label_b)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax.grid()
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
        print(fname)
    plt.close()


def load_data(filename):
    raw = np.loadtxt(filename)
    ckpt = np.unique(raw[:, 0])
    it = np.unique(raw[:, 1])
    num_eval = it.shape[0]
    num_ckpt = ckpt.shape[0]
    data = np.empty((num_ckpt, raw.shape[1]))
    for row in range(num_ckpt):
        COL_FID = 3
        idx_fid = np.argmin(raw[row*num_eval:row*num_eval+num_eval-1, COL_FID])
        data[row] = raw[row*num_eval+idx_fid]
    print("filename", filename)
    print("data", data)
    return raw, data


def compare_discriminator_update(args, logdir, raw_standard, raw_rejection, raw_hastings, raw_refinement, raw_collaborate):
    num_eval = np.unique(raw_standard[:, 1]).shape[0]
    num_ckpt = np.unique(raw_standard[:, 0]).shape[0]
    for idx_ckpt in range(num_ckpt):
        if args.mode == "calibrate":
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2], raw_rejection[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2], raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2],
                'standard', 'rejection', 'mh',
                'D Iteration', 'CS', fname=logdir + args.mode + "_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_is.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3], raw_rejection[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3], raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3],
                'standard', 'rejection', 'mh',
                'D Iteration', 'FD', fname=logdir + args.mode + "_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_fid.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4], raw_rejection[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4], raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4],
                'standard', 'rejection', 'mh',
                'D Iteration', 'Efficiency', fname=logdir + args.mode + "_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_eff.png")
        if args.mode == "shape":
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2], raw_refinement[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2], raw_collaborate[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2],
                'standard', 'refinement', 'collaborative',
                'D Iteration', 'CS', fname=logdir + args.mode + "_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_is.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3], raw_refinement[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3], raw_collaborate[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3],
                'standard', 'refinement', 'collaborative',
                'D Iteration', 'FD', fname=logdir + args.mode + "_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_fid.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4], raw_refinement[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4], raw_collaborate[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4],
                'standard', 'refinement', 'collaborative',
                'D Iteration', 'Efficiency', fname=logdir + args.mode + "_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_eff.png")

def compare_generator_stage(args, logdir, data_standard, data_rejection, data_hastings, data_refinement, data_collaborate):
    plot_method(data_standard[1:, 0], data_standard[1:, 2], data_rejection[1:, 2], data_hastings[1:, 2], data_refinement[1:, 2], data_collaborate[1:, 2],
        'standard', 'rejection', 'mh', 'refinement', 'collaborative',
        'GAN Iteration', 'CS', fname=logdir + args.mode + "_is.png")
    plot_method(data_standard[1:, 0], data_standard[1:, 3], data_rejection[1:, 3], data_hastings[1:, 3], data_refinement[1:, 3], data_collaborate[1:, 3],
        'standard', 'rejection', 'mh', 'refinement', 'collaborative',
        'GAN Iteration', 'FD', fname=logdir + args.mode + "_fd.png")


def main():
    args = parse_args()
    logdir = args.logdir
    if args.mode == "benchmark":
        raw_standard, data_standard = load_data(logdir + "shape_standard.txt")
        raw_rejection, data_rejection = load_data(logdir + "calibrate_rejection.txt")
        raw_hastings, data_hastings = load_data(logdir + "calibrate_hastings.txt")
        raw_refinement, data_refinement = load_data(logdir + "shape_refinement.txt")
        raw_collaborate, data_collaborate = load_data(logdir + "shape_collaborate.txt")
    elif args.mode == "calibrate":
        raw_standard, data_standard = load_data(logdir + args.mode + "_standard.txt")
        raw_rejection, data_rejection = load_data(logdir + args.mode + "_rejection.txt")
        raw_hastings, data_hastings = load_data(logdir + args.mode + "_hastings.txt")
        raw_refinement, raw_collaborate = [], []
    elif args.mode == "shape":
        raw_standard, data_standard = load_data(logdir + args.mode + "_standard.txt")
        raw_refinement, data_refinement = load_data(logdir + args.mode + "_refinement.txt")
        raw_collaborate, data_collaborate = load_data(logdir + args.mode + "_collaborate.txt")
        raw_rejection, raw_hastings = [], []

    if args.mode == "benchmark":
        compare_generator_stage(args, logdir, data_standard, data_rejection, data_hastings, data_refinement, data_collaborate)
    else:
        compare_discriminator_update(args, logdir, raw_standard, raw_rejection, raw_hastings, raw_refinement, raw_collaborate)


if __name__ == '__main__':
    main()
