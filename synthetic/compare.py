import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr, spearmanr
np.set_printoptions(suppress=True, precision=2)

"""parsing and configuration"""
def parse_args():
    desc = "Compare various sampling methods for GANs"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Imbal-8Gaussians',
                        help='dataset to use: 25Gaussians | Imbal-8Gaussians')
    parser.add_argument('--mode', type=str, default='benchmark',
                        help='type of running: train, shape, calibrate, benchmark, diagnostic')
    return parser.parse_args()

def normalized(x):
    return (x-min(x))/(max(x)-min(x))

def plot_method(x, y_r, y_g, y_b, y_c, y_k, label_r, label_g, label_b, label_c, label_k, label_x, label_y, ylim=None, fname=None):
    fig = plt.figure(figsize=(4, 3))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    ax.plot(x, y_r, 'r-x', label=label_r, linewidth=1.5)
    ax.plot(x, y_g, 'c-+', label=label_g, linewidth=1.5)
    ax.plot(x, y_b, 'b-1', label=label_b, linewidth=1.5)
    ax.plot(x, y_c, 'g-o', label=label_c, linewidth=1.5)
    ax.plot(x, y_k, 'k-s', label=label_k, linewidth=1.5)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax.grid()
    # plt.legend(loc=3,ncol=5,bbox_to_anchor=(0.0, 1.05))
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
        print(fname)


def plot_mode(x, y_r, y_g, y_b, label_r, label_g, label_b, label_x, label_y, ylim=None, fname=None):
    fig = plt.figure(figsize=(4, 3))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    ax.plot(x, y_r, 'r-x', label=label_r, linewidth=2, markersize=12)
    ax.plot(x, y_g, 'y-o', label=label_g, linewidth=2, markersize=12)
    ax.plot(x, y_b, 'b-*', label=label_b, linewidth=2, markersize=12)
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

def plot_diagnostic(x, y_s, y_z, y_c, y_b, fname):
    fig = plt.figure(figsize=(4, 3))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    xmax = 11 # x range
    ax.plot(x[:xmax], normalized(y_s[:xmax]), 'b-', label='JS', linewidth=1.5, markersize=12)
    ax.plot(x[:xmax], normalized(abs(y_z[:xmax])), 'r-', label='|Z|', linewidth=1.5, markersize=12)
    ax.plot(x[:xmax], normalized(y_c[:xmax]), 'g-', label='ECE', linewidth=1.5, markersize=12)
    ax.plot(x[:xmax], normalized(y_b[:xmax]), 'k-', label='BS', linewidth=1.5, markersize=12)
    ax.set_xlabel('D Iteration')
    ax.set_ylabel('Normalized Metric')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax.grid()
    plt.ylim([0.0, 1.0])
    plt.legend(loc='upper right')
    if fname:
        plt.savefig(fname, bbox_inches='tight')
        print(fname)

def load_data(filename):
    raw = np.loadtxt(filename)
    it = np.unique(raw[:, 1])
    num_eval = it.shape[0]
    num_avg = 4
    data = raw[num_eval-1::num_eval, :].copy()
    for i in range(1, num_avg):
        data += raw[num_eval-num_avg-1::num_eval, :]
    data /= num_avg
    return raw, data


def compare_discriminator_update(args,logdir, raw_standard, raw_rejection, raw_hastings, raw_refinement, raw_collaborate):
    num_eval = np.unique(raw_standard[:, 1]).shape[0]
    num_ckpt = np.unique(raw_standard[:, 0]).shape[0]
    for idx_ckpt in range(num_ckpt):
        if args.mode == "calibrate":
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2], raw_rejection[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2], raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2],
                        'standard', 'rejection', 'mh',
                        'D Iteration', 'Mean Distance', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_distance.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3], raw_rejection[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3], raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3],
                        'standard', 'rejection', 'mh',
                        'D Iteration', 'Good Samples', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_good.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4], raw_rejection[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4], raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4],
                        'standard', 'rejection', 'mh',
                        'D Iteration', 'KL Divergence', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_kl.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 5], raw_rejection[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 5], raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 5],
                        'standard', 'rejection', 'mh',
                        'D Iteration', 'JS Divergence', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_js.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 6], raw_rejection[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 6], raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 6],
                        'standard', 'rejection', 'mh',
                        'D Iteration', 'Efficiency', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_eff.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 7], raw_rejection[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 7], raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 7],
                        'standard', 'rejection', 'mh',
                        'D Iteration', 'Diagnostic', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_diagnostic.png")
            plot_diagnostic(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1],
                            raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 5],
                            raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 7],
                            raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 9],
                            raw_hastings[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 8],
                            fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_metrics.png"
                            )

        if args.mode == "shape":
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2], raw_refinement[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2], raw_collaborate[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 2],
                        'standard', 'refinement', 'collaborative',
                        'D Iteration', 'Mean Distance', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_distance.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3], raw_refinement[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3], raw_collaborate[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 3],
                        'standard', 'refinement', 'collaborative',
                        'D Iteration', 'Good Samples', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_good.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4], raw_refinement[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4], raw_collaborate[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 4],
                        'standard', 'refinement', 'collaborative',
                        'D Iteration', 'KL Divergence', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_kl.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 5], raw_refinement[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 5], raw_collaborate[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 5],
                        'standard', 'refinement', 'collaborative',
                        'D Iteration', 'JS Divergence', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_js.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 6], raw_refinement[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 6], raw_collaborate[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 6],
                        'standard', 'refinement', 'collaborative',
                        'D Iteration', 'Efficiency', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_eff.png")
            plot_mode(raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 1], raw_standard[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 7], raw_refinement[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 7], raw_collaborate[idx_ckpt*num_eval:(idx_ckpt+1)*num_eval, 7],
                        'standard', 'refinement', 'collaborative',
                        'D Iteration', 'Diagnostic', fname=logdir + "_" + args.mode + "/2d_G_" + "{:d}".format(int(raw_standard[idx_ckpt*num_eval, 0])) + "_diagnostic.png")


def compare_generator_stage(args, logdir, data_standard, data_rejection, data_hastings, data_refinement, data_collaborate):

    plot_method(data_standard[:, 0], data_standard[:, 2], data_rejection[:, 2], data_hastings[:, 2], data_refinement[:, 2], data_collaborate[:, 2],
               'standard', 'rejection', 'mh', 'refinement', 'collaborative',
               'GAN Iteration', 'Mean Distance', fname=logdir + "_" + args.mode + "_2d_distance.png")
    plot_method(data_standard[:, 0], data_standard[:, 3], data_rejection[:, 3], data_hastings[:, 3], data_refinement[:, 3], data_collaborate[:, 3],
               'standard', 'rejection', 'mh', 'refinement', 'collaborative',
               'GAN Iteration', 'Good Samples', fname=logdir + "_" + args.mode + "_2d_good.png")
    plot_method(data_standard[:, 0], data_standard[:, 4], data_rejection[:, 4], data_hastings[:, 4], data_refinement[:, 4], data_collaborate[:, 4],
               'standard', 'rejection', 'mh', 'refinement', 'collaborative',
               'GAN Iteration', 'KL Divergence', fname=logdir + "_" + args.mode + "_2d_kl.png")
    plot_method(data_standard[:, 0], data_standard[:, 5], data_rejection[:, 5], data_hastings[:, 5], data_refinement[:, 5], data_collaborate[:, 5],
               'standard', 'rejection', 'mh', 'refinement', 'collaborative',
               'GAN Iteration', 'JS Divergence', fname=logdir + "_" + args.mode + "_2d_js.png")
    plot_method(data_standard[:, 0], data_standard[:, 6], data_rejection[:, 6], data_hastings[:, 6], data_refinement[:, 6], data_collaborate[:, 6],
               'standard', 'rejection', 'mh', 'refinement', 'collaborative',
               'GAN Iteration', 'Sample Efficiency', fname=logdir + "_" + args.mode + "_2d_eff.png")

def plot_attribute_diagnostic(raw_attribute, diff_method, attribute_name, num_eval, num_ckpt, fname):

    fig = plt.figure(figsize=(10, 10))  # create a figure object

    ax1 = fig.add_subplot(2, 2, 1)  # create an axes object in the figure
    corr_spearman_sum, p_spearman_sum = 0.0, 0.0
    corr_pearson_sum, p_pearson_sum = 0.0, 0.0
    for i in range(num_ckpt):
        ax1.scatter(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 2], marker=i, s=50)
        corr_spearman, p_spearman = spearmanr(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 2])
        corr_pearson, p_pearson = pearsonr(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 2])
        corr_spearman_sum += corr_spearman
        p_spearman_sum += p_spearman
        corr_pearson_sum += corr_pearson
        p_pearson_sum += p_pearson
    ax1.set_title('spearman: {:.2f} ({:.2f}),  pearson: {:.2f} ({:.2f})'.format(corr_spearman_sum / num_ckpt, p_spearman_sum / num_ckpt, corr_pearson_sum / num_ckpt, p_pearson_sum / num_ckpt))
    ax1.set_xlabel(attribute_name)
    ax1.set_ylabel('dist')
    ax1.grid()

    ax2 = fig.add_subplot(2, 2, 2)  # create an axes object in the figure
    corr_spearman_sum, p_spearman_sum = 0.0, 0.0
    corr_pearson_sum, p_pearson_sum = 0.0, 0.0
    for i in range(num_ckpt):
        ax2.scatter(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 3], marker=i, s=50)
        corr_spearman, p_spearman = spearmanr(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 3])
        corr_pearson, p_pearson = pearsonr(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 3])
        corr_spearman_sum += corr_spearman
        p_spearman_sum += p_spearman
        corr_pearson_sum += corr_pearson
        p_pearson_sum += p_pearson
    ax2.set_title('spearman: {:.2f} ({:.2f}),  pearson: {:.2f} ({:.2f})'.format(corr_spearman_sum / num_ckpt, p_spearman_sum / num_ckpt, corr_pearson_sum / num_ckpt, p_pearson_sum / num_ckpt))
    ax2.set_xlabel(attribute_name)
    ax2.set_ylabel('good')
    ax2.grid()

    ax3 = fig.add_subplot(2, 2, 3)  # create an axes object in the figure
    corr_spearman_sum, p_spearman_sum = 0.0, 0.0
    corr_pearson_sum, p_pearson_sum = 0.0, 0.0
    for i in range(num_ckpt):
        ax3.scatter(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 4], marker=i, s=50)
        corr_spearman, p_spearman = spearmanr(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 4])
        corr_pearson, p_pearson = pearsonr(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 4])
        corr_spearman_sum += corr_spearman
        p_spearman_sum += p_spearman
        corr_pearson_sum += corr_pearson
        p_pearson_sum += p_pearson
    ax3.set_title('spearman: {:.2f} ({:.2f}),  pearson: {:.2f} ({:.2f})'.format(corr_spearman_sum / num_ckpt, p_spearman_sum / num_ckpt, corr_pearson_sum / num_ckpt, p_pearson_sum / num_ckpt))
    ax3.set_xlabel(attribute_name)
    ax3.set_ylabel('kl')
    ax3.grid()

    ax4 = fig.add_subplot(2, 2, 4)  # create an axes object in the figure
    corr_spearman_sum, p_spearman_sum = 0.0, 0.0
    corr_pearson_sum, p_pearson_sum = 0.0, 0.0
    for i in range(num_ckpt):
        ax4.scatter(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 5], marker=i, s=50)
        corr_spearman, p_spearman = spearmanr(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 5])
        corr_pearson, p_pearson = pearsonr(raw_attribute[i*num_eval:i*num_eval+num_eval], diff_method[i*num_eval:i*num_eval+num_eval, 5])
        corr_spearman_sum += corr_spearman
        p_spearman_sum += p_spearman
        corr_pearson_sum += corr_pearson
        p_pearson_sum += p_pearson
    ax4.set_title('spearman: {:.2f} ({:.2f}),  pearson: {:.2f} ({:.2f})'.format(corr_spearman_sum / num_ckpt, p_spearman_sum / num_ckpt, corr_pearson_sum / num_ckpt, p_pearson_sum / num_ckpt))
    ax4.set_xlabel(attribute_name)
    ax4.set_ylabel('js')
    ax4.grid()

    plt.savefig(fname + attribute_name + '.png', bbox_inches='tight')


def plot_discriminator_diagnostic(raw_standard, raw_method, fname):
    num_eval = np.unique(raw_standard[:, 1]).shape[0]
    num_ckpt = np.unique(raw_standard[:, 0]).shape[0]
    diff_method = raw_method - raw_standard

    plot_attribute_diagnostic(raw_method[:, 1], diff_method, 'iteraction', num_eval, num_ckpt, fname)
    plot_attribute_diagnostic(abs(raw_method[:, 7]), diff_method, 'abs(Z)', num_eval, num_ckpt, fname)
    plot_attribute_diagnostic(raw_method[:, 8], diff_method, 'BS', num_eval, num_ckpt, fname)
    plot_attribute_diagnostic(raw_method[:, 9], diff_method, 'ECE', num_eval, num_ckpt, fname)
    plot_attribute_diagnostic(raw_method[:, 10], diff_method, 'MCE', num_eval, num_ckpt, fname)


def compare_discriminator_diagnostic(logdir, raw_standard, raw_rejection, raw_hastings):
    print("rejection diagnostic")
    plot_discriminator_diagnostic(raw_standard, raw_rejection, logdir + '_diagnostic_reject_')
    print("mh diagnostic")
    plot_discriminator_diagnostic(raw_standard, raw_hastings, logdir + '_diagnostic_mh_')


def main():
    args = parse_args()
    logdir = "log/" + args.dataset

    if args.mode == "calibrate":
        raw_standard, data_standard = load_data(logdir + "_" + args.mode + "/" + args.mode + "_standard.txt")
        raw_rejection, data_rejection = load_data(logdir + "_" + args.mode + "/" + args.mode + "_rejection.txt")
        raw_hastings, data_hastings = load_data(logdir + "_" + args.mode + "/" + args.mode + "_hastings.txt")
        raw_refinement, raw_collaborate = [], []
    elif args.mode == "shape":
        raw_standard, data_standard = load_data(logdir + "_" + args.mode + "/" + args.mode + "_standard.txt")
        raw_refinement, data_refinement = load_data(logdir + "_" + args.mode + "/" + args.mode + "_refinement.txt")
        raw_collaborate, data_collaborate = load_data(logdir + "_" + args.mode + "/" + args.mode + "_collaborate.txt")
        raw_rejection, raw_hastings = [], []
    else:
        raw_standard, data_standard = load_data(logdir + "_calibrate/calibrate_standard.txt")
        raw_rejection, data_rejection = load_data(logdir + "_calibrate/calibrate_rejection.txt")
        raw_hastings, data_hastings = load_data(logdir + "_calibrate/calibrate_hastings.txt")
        raw_refinement, data_refinement = load_data(logdir + "_shape/shape_refinement.txt")
        raw_collaborate, data_collaborate = load_data(logdir + "_shape/shape_collaborate.txt")

    if args.mode == "benchmark":
        compare_generator_stage(args, logdir, data_standard, data_rejection, data_hastings, data_refinement, data_collaborate)
    elif args.mode == "diagnostic":
        compare_discriminator_diagnostic(logdir, raw_standard, raw_rejection, raw_hastings)
    else:
        compare_discriminator_update(args, logdir, raw_standard, raw_rejection, raw_hastings, raw_refinement, raw_collaborate)


if __name__ == '__main__':

    # main functions
    main()
