from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import brier_score_loss

def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def draw_sample(model_batch, real_batch, scale, fname, color=None):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(1, 1, 1)
    if real_batch is not None:
        ax.scatter(real_batch[:, 0], real_batch[:, 1], s=100, c='g', alpha=0.1)
    if color is None:
        color = 'b'
    if model_batch is not None:
        ax.scatter(model_batch[:, 0], model_batch[:, 1], s=100, c=color, alpha=0.1)
    ax.set_xlim((-scale, scale))
    ax.set_ylim((-scale, scale))
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw_landscape(grid_batch, grid_sigmoid, real_batch, scale, fname):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(1, 1, 1)

    # Discriminator contour
    x_mesh = np.reshape(grid_batch[:, 0], [int(np.sqrt(grid_batch.shape[0])), -1]).T
    y_mesh = np.reshape(grid_batch[:, 1], [int(np.sqrt(grid_batch.shape[0])), -1]).T
    v_mesh = np.reshape(grid_sigmoid, [int(np.sqrt(grid_batch.shape[0])), -1]).T
    ax.contourf(x_mesh, y_mesh, v_mesh, 100, cmap='Greys', vmin=0.0, vmax=0.7)

    # Real samples
    ax.scatter(real_batch[:, 0], real_batch[:, 1], s=10, c='g')

    ax.set_xlim((-scale, scale))
    ax.set_ylim((-scale, scale))
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw_density(samps, scale, fname):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(1, 1, 1)

    # Real samples
    import seaborn as sns
    sns.kdeplot(samps[:, 0], samps[:, 1], shade=True, cmap='Greys', n_levels=100)

    ax.set_xlim((-scale, scale))
    ax.set_ylim((-scale, scale))
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw_histogram(samps, scale, fname):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(1, 1, 1)

    ax.hist2d(samps[:, 0], samps[:, 1], bins=(50, 50), cmap=plt.cm.BuPu)
    ax.set_xlim((-scale, scale))
    ax.set_ylim((-scale, scale))
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw_kde(samps, scale, fname):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(1, 1, 1)

    from scipy.stats import kde
    nbins = 100
    x = samps[:, 0]
    y = samps[:, 1]
    k = kde.gaussian_kde([x, y])
    k.set_bandwidth(bw_method=k.factor/2.)
    xi, yi = np.mgrid[-scale:scale:nbins*1j, -scale:scale:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    vmax_factor = 0.2
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuPu, vmin=np.min(zi), vmax=max(np.max(zi)*vmax_factor, np.min(zi)))

    ax.set_xlim((-scale, scale))
    ax.set_ylim((-scale, scale))
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def pairwise_distance(real_batch, model_batch):
    dist_matrix = cdist(real_batch, model_batch)
    dist_eye = 10.0*np.identity(dist_matrix.shape[0])
    dist_min = np.min(dist_matrix+dist_eye, axis=0)
    return dist_min

def metrics_distance(samples, centeroids, thres):
    samples = np.array(samples)
    centeroids = np.array(centeroids)
    n = np.size(samples, 0)
    k = np.size(centeroids, 0)
    distances = np.zeros((n, k))
    for i in range(k):
        distances[:, i] = np.linalg.norm(samples - centeroids[i], axis=1)
    dist_min = np.min(distances, axis=1)
    cnt_good = (dist_min < thres).sum()
    cnt_all = dist_min.size
    rate_good = cnt_good / cnt_all
    mean_dist = np.mean(dist_min)
    return mean_dist, rate_good

def freq_category(samples, centeroids, thres):
    n_category = np.size(centeroids, 0)
    n_samples = np.size(samples, 0)
    distances = np.ones((n_samples, n_category))
    for i in range(n_category):
        distances[:, i] = np.linalg.norm(samples - centeroids[i], axis=1)
    clusters = distances < thres
    counts = np.sum(clusters, axis=0)
    sum_counts = np.sum(counts)
    if sum_counts > 0:
        freqs_valid = counts / sum_counts
    else:
        freqs_valid = np.ones(n_category) / n_category  # uniform
    freqs_all = np.append(counts, n_samples-sum_counts) / n_samples
    return freqs_valid, freqs_all

def metrics_diversity(real_batch, model_batch, centeroids, thres):
    freq_real, _ = freq_category(real_batch, centeroids, thres)
    freq_model, _ = freq_category(model_batch, centeroids, thres)
    kl = kl_div(freq_model, freq_real)
    return kl

def kl_div(predictions, targets):
    """
    Input: predictions (k,1) ndarray
           targets (k,1) ndarray
    Returns: scalar
    """
    targets = np.clip(targets, a_min=1e-12, a_max=1 - 1e-12)
    kl = stats.entropy(predictions, targets)
    return kl

def metrics_distribution(real_batch, model_batch, centeroids, thres):
    _, freq_real = freq_category(real_batch, centeroids, thres)
    _, freq_model = freq_category(model_batch, centeroids, thres)
    freq_avg = (freq_real + freq_model) * 0.5
    js = 0.5 * kl_div(freq_real, freq_avg) + 0.5 * kl_div(freq_model, freq_avg)
    return js

def calib_score(y_prob, y_true):
    '''
    Function borrowed from MH-GAN
    y_prob : ndarray, shape (n,)
        floats in [0, 1]
    y_true : ndarray, shape (n,)
        bool
    '''
    Z = np.sum(y_true - y_prob) / np.sqrt(np.sum(y_prob * (1.0 - y_prob)))
    return Z

def column_or_1d(y, warn=False):
    """ Ravel column or 1d numpy array, else raises an error
    Parameters
    ----------
    y : array-like
    warn : boolean, default False
       To control display of warnings.
    Returns
    -------
    y : array
    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            pass
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


def calibration_bins(y_true, y_prob, n_bins=50, strategy='uniform'):
    """Compute true and predicted probabilities for a calibration curve.
    The method assumes the inputs come from a binary classifier.
    Calibration curves may also be referred to as reliability diagrams.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    normalize : bool, optional, default=False
        Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
        a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.
    n_bins : int
        Number of bins. A bigger number requires more data. Bins with no data
        points (i.e. without corresponding values in y_prob) will not be
        returned, thus there may be fewer than n_bins in the return value.
    strategy : {'uniform', 'quantile'}, (default='uniform')
        Strategy used to define the widths of the bins.
        uniform
            All bins have identical widths.
        quantile
            All bins have the same number of points.
    Returns
    -------
    prob_true : array, shape (n_bins,) or smaller
        The true probability in each bin (fraction of positives).
    prob_pred : array, shape (n_bins,) or smaller
        The mean predicted probability in each bin.
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)

    if strategy == 'quantile':  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == 'uniform':
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy "
                         "must be either 'quantile' or 'uniform'.")

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    weights_bin = bin_total[nonzero] / y_true.shape[0]

    return prob_true, prob_pred, weights_bin

def calibration_error(confidence, accuracy, weights):
    assert np.absolute(np.sum(weights)-1.0) < 1e-4, 'Abnormal sum of weights, residual = {:.6f}'.format((np.sum(weights)-1.0))
    expeced_calibration_error = np.sum(np.absolute(accuracy - confidence) * weights)
    maximum_calibration_error = np.max(np.absolute(accuracy - confidence))
    return expeced_calibration_error, maximum_calibration_error

def calibration_diagnostic(fake_sigmoid, real_sigmoid, fname=None):
    y_pred = np.concatenate([(fake_sigmoid), (real_sigmoid)])
    y_true = np.concatenate([np.zeros_like(fake_sigmoid), np.ones_like(real_sigmoid)])
    accuracy, confidence, weights = calibration_bins(y_true, y_pred, n_bins=20)
    z_dawid = calib_score(y_pred, y_true)
    brier_score = brier_score_loss(y_true, y_pred)
    ece, mce = calibration_error(confidence, accuracy, weights)
    if fname is not None:
        draw_reliability(confidence, accuracy, weights, z_dawid, brier_score, ece, mce, fname)
    return z_dawid, brier_score, ece, mce

def draw_reliability(confidence, accuracy, weights, dawid, brier, ece, mce, fname):
    fig = plt.figure(frameon=False, figsize=(4.6, 6))
    gridspec.GridSpec(5, 1)
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    ax2 = plt.subplot2grid((5, 1), (4, 0))

    ax1.plot([0, 1], [0, 1], "k:")
    ax1.plot(confidence, accuracy, "s-")
    ax1.set_xlim((0.0, 1.0))
    ax1.set_ylim((0.0, 1.0))
    ax1.set_aspect('equal')
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.set_ylabel("Accuracy")
    ax1.set_title('dawid: {:.2f}, brier: {:.2f}, ece: {:.2f}, mce: {:.2f}'.format(dawid, brier, ece, mce))

    ax2.plot(confidence, weights, 'x')
    ax2.set_xlim((0.0, 1.0))
    ax2.set_ylim((0.0, 0.6))
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Weights")

    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
