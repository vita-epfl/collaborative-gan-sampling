from __future__ import division
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pickle
import os 
import shutil
import time
from scipy import stats

np.random.seed(1234)

from operator import itemgetter
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

############################################################
####### GAN Viz ############################################
############################################################
def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def plot_loss(prefix, norm_grad_list, g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list, teacher_name = None, rollout_method = None, rollout_steps = None, rollout_rate = None):
    f, ax = plt.subplots(1)
    WINDOW = max(1,min(100, len(g_loss_list)-5))
    ALPHA = 0.3 
    if len(norm_grad_list):
        norm_grad_array = np.array(norm_grad_list)
        ax.semilogy(norm_grad_array[:,0], norm_grad_array[:,1], '--', color="k", label='grad_norm_median')
    if len(g_loss_list):
        g_loss_array = np.array(g_loss_list)
        ax.semilogy(g_loss_array[:,0], g_loss_array[:,1], color="m", alpha=ALPHA)
        ax.semilogy(g_loss_array[WINDOW-1:,0], moving_average(g_loss_array[:,1],WINDOW), color="m", label='g_loss')
    if len(d_loss_list):
        d_loss_array = np.array(d_loss_list)
        ax.semilogy(d_loss_array[:,0], d_loss_array[:,1], color="r", alpha=ALPHA)
        ax.semilogy(d_loss_array[WINDOW-1:,0], moving_average(d_loss_array[:,1],WINDOW), color="r", label='d_loss')
    if len(d_loss_fake_list):
        d_loss_fake_array = np.array(d_loss_fake_list)
        ax.semilogy(d_loss_fake_array[:,0], d_loss_fake_array[:,1], color="g", alpha=ALPHA)
        ax.semilogy(d_loss_fake_array[WINDOW-1:,0], moving_average(d_loss_fake_array[:,1],WINDOW), color="g", label='d_loss_fake')
    if len(d_loss_real_list):
        d_loss_real_array = np.array(d_loss_real_list)
        ax.semilogy(d_loss_real_array[:,0], d_loss_real_array[:,1], color="b", alpha=ALPHA)
        ax.semilogy(d_loss_real_array[WINDOW-1:,0], moving_average(d_loss_real_array[:,1],WINDOW), color="b", label='d_loss_real')
    ax.set_ylim(0.05, 10.0)

    ax.grid(True)
    
    title = teacher_name
    if teacher_name == 'rollout':
        title += ': ' + rollout_method + ', step={:d}, rate={:.1e}'.format(rollout_steps,rollout_rate)
    ax.set_title(title)

    plt.xlabel('Step')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(prefix + 'metrics.png')

def plot_samples(prefix, ax, scale, real_batch, grid_batch, grid_grad, grid_sigmoid, it, loss, norm_grad, fake_batch=None, teacher_batch=None, perturbed_batch=None, xmin=None, xmax=None, ymin=None, ymax=None, grid_grad_teacher=None, title=None):
    # print("Iter: %d, loss: %.8f, norm_grad = %.8f" % (it, loss, norm_grad))
    
    ax.clear()

    # Discriminator contour
    if grid_batch is not None:
        x_mesh = np.reshape(grid_batch[:,0],[int(np.sqrt(grid_batch.shape[0])),-1]).T
        y_mesh = np.reshape(grid_batch[:,1],[int(np.sqrt(grid_batch.shape[0])),-1]).T
        v_mesh = np.reshape(grid_sigmoid,[int(np.sqrt(grid_batch.shape[0])),-1]).T
        ax.contourf(x_mesh, y_mesh, v_mesh, 50, cmap='Greys', vmin = 0.2, vmax = 0.8)

    # norm_grad_mean = np.mean(np.linalg.norm(grid_grad_teacher, axis=1))
    # norm_grad_mean = (np.mean(np.linalg.norm(grid_grad, axis=1)) + np.mean(np.linalg.norm(grid_grad_teacher, axis=1))) / 2
    # , scale=norm_grad_mean*10

        ax.quiver(grid_batch[:, 0], grid_batch[:, 1], -grid_grad[:, 0], -grid_grad[:, 1], color='b')
        if grid_grad_teacher is not None:
            ax.quiver(grid_batch[:, 0], grid_batch[:, 1], -grid_grad_teacher[:, 0], -grid_grad_teacher[:, 1], color='g')

    # sample distribution
    if perturbed_batch is not None:
        ax.scatter(perturbed_batch[:, 0], perturbed_batch[:, 1], s=10, c='c', label='perturb')        
    if real_batch is not None:
        ax.scatter(real_batch[:, 0], real_batch[:, 1], s=10, c='y', label='real')
    if teacher_batch is not None:
        ax.scatter(teacher_batch[:, 0], teacher_batch[:, 1], s=10, c='g', label='target')
    if fake_batch is not None:
        ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=10, c='r', label='fake')

    # if xmin.size and xmax.size and ymin.size and ymax.size:
    #     ax.set_xlim((xmin,xmax))
    #     ax.set_ylim((ymin,ymax))
    # else:
    ax.set_xlim((-scale*1.3, scale*1.3))
    ax.set_ylim((-scale*1.3, scale*1.3))

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if title is None:
        title = "Iter #{}".format(it)
    ax.set_title(title)
    
    ax.legend()
    plt.draw()
    # plt.show()

def plot_image(image, ax, it, desc):    
    ax.clear()

    ax.imshow(image, vmin=-1.0, vmax=1.0)

    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    ax.set_title(desc + " Iter #{:d}".format(it))

    # plt.legend()
    plt.draw()
    # plt.show()

def plot_trajectories(ax, x_traj, y_traj, grid_batch, grid_sigmoid, mean_sigmoid, indices_optimal):

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.clear()

    # Discriminator contour
    x_mesh = np.reshape(grid_batch[:,0],[int(np.sqrt(grid_batch.shape[0])),-1]).T
    y_mesh = np.reshape(grid_batch[:,1],[int(np.sqrt(grid_batch.shape[0])),-1]).T
    v_mesh = np.reshape(grid_sigmoid,[int(np.sqrt(grid_batch.shape[0])),-1]).T
    ax.contourf(x_mesh, y_mesh, v_mesh, 50, cmap='Greys')
    
    idx_rand = np.random.randint(len(x_traj), size=5)
    for idx in idx_rand:
        ax.plot(x_traj[idx,0], y_traj[idx,0], 'ro')
        ax.plot(x_traj[idx,:], y_traj[idx,:], 'b-')
        ax.plot(x_traj[idx,-1], y_traj[idx,-1], 'gs')
        ax.plot(x_traj[idx,indices_optimal[idx]], y_traj[idx,indices_optimal[idx]], 'c^')

    ax.set_title(str(mean_sigmoid))

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # prefix = 'figs/rollout/' 
    # plt.savefig(prefix + 'rollout_%.4f.png' % mean_sigmoid, bbox_inches='tight')
    plt.draw()

def plot_norm_distribution(ax, loss, grad_default, grad_teacher, teacher_name, rollout_method, rollout_steps, rollout_rate):

    UB_NORM = 1.0
    LB_NORM = 1e-5    

    ax.clear()

    if grad_default.ndim > 2:
        grad_default = np.reshape(grad_default,(grad_default.shape[0],-1))

    norm_grad_default = np.linalg.norm(grad_default, axis=1)

    ax.semilogy(np.squeeze(loss), norm_grad_default.clip(min=LB_NORM,max=UB_NORM), 'bo', label='default')

    if grad_teacher is not None:
        if grad_teacher.ndim > 2:
            grad_teacher = np.reshape(grad_teacher,(grad_teacher.shape[0],-1))
        norm_grad_teacher = np.linalg.norm(grad_teacher, axis=1)
        ax.semilogy(np.squeeze(loss), norm_grad_teacher.clip(min=LB_NORM,max=UB_NORM), 'gx', label='teacher')

    ax.set_xlabel("loss\' = $1 - D(G(z))$")
    ax.set_ylabel("grad norm")
    ax.set_xlim((0.0,1.0))
    ax.set_ylim((LB_NORM, UB_NORM))

    title = teacher_name
    if teacher_name == 'rollout':
        title += ': ' + rollout_method + ', step={:d}, rate={:.1e}'.format(rollout_steps,rollout_rate)
    ax.set_title(title)

    plt.legend()

    plt.draw()
    plt.pause(1e-6)
    # plt.show()


def plot_norm_histogram(ax, loss, grad_default, grad_teacher, teacher_name, rollout_method, rollout_steps, rollout_rate):

    ax.clear()

    # the histogram of the data
    n, bins, patches = plt.hist(loss, bins=50, density=1, facecolor='green')

    ax.set_xlabel("$\Delta S$")
    ax.set_ylabel("Probability")
    ax.set_xlim((0.0,1.0))
    ax.set_ylim((0.0,20.0))

    ax.grid(True)

    title = teacher_name
    if teacher_name == 'rollout':
        title += ': ' + rollout_method + ', step={:d}, rate={:.1e}'.format(rollout_steps,rollout_rate)
    ax.set_title(title)

    plt.legend()

    plt.draw()
    # plt.pause(1e-6)
    plt.show()

def plot_figs(grad_default, D_sigmoid_fake, grad_default_real, D_sigmoid_real, batch_size, optimal_batch, input_batch, optimal_grad, counter, epoch, idx, config):
    
    fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(9, 9))
    fig3, (ax_31, ax_32) = plt.subplots(1, 2, figsize=(8, 8))
    min_ind = np.argmin(D_sigmoid_fake)
    y_axis = np.linalg.norm(np.reshape(grad_default[0][0], (batch_size, -1)), axis=1)
    x_axis = np.squeeze(1 - D_sigmoid_fake)
    y_axis2 = np.linalg.norm(np.reshape(grad_default_real[0][0], (batch_size, -1)), axis=1)
    x_axis2 = np.squeeze(1 - D_sigmoid_real)
    plt.figure(fig3.number)
    ax_31.clear()
    ax_32.clear()
    ax_31.scatter(x_axis, y_axis)
    ax_31.scatter(x_axis2, y_axis2)
    ax_31.set_xlim(left=0.0, right=1.0)
    ax_31.set_ylim(bottom=0.0)
    ax_32.hist(y_axis, bins=10)
    ax_32.hist(y_axis2, bins=10)
    plt.draw()
    # plt.pause(1e-4)
    # plt.show()
    if config.save_figs:
      fig3.savefig('./{}/viz_grad_{:02d}_{:04d}.png'.format('figs', epoch, idx), bbox_inches='tight')
    # print(np.max(np.squeeze(optimal_grad[0] - grad_default[0][0][0])), np.min(np.squeeze(optimal_grad[0] - grad_default[0][0][0])))
    optimal_batch = 0.5*(optimal_batch + 1) 
    input_batch = 0.5*(input_batch + 1)
    plot_image(np.squeeze(optimal_batch[min_ind]), ax1, counter, desc="Optimal Image")
    plot_image(np.squeeze(input_batch[min_ind]), ax2, counter, desc="Default Image")
    plot_image(np.squeeze(optimal_batch[min_ind] - input_batch[min_ind]), ax3, counter, desc="Difference between Image")
    plot_image(np.squeeze(optimal_grad[min_ind]), ax4, counter, desc="Optimal Grad")    
    plot_image(np.squeeze(grad_default[0][0][min_ind]), ax5, counter, desc="Grad Default")                    
    plot_image(np.squeeze(optimal_grad[min_ind] - grad_default[0][0][min_ind]), ax6, counter, desc="Difference between Grads")            
    plt.figure(fig1.number)
    plt.draw()
    plt.show()
    if config.save_figs:
      fig1.savefig('./{}/viz_image_{:02d}_{:04d}.png'.format('figs', epoch, idx), bbox_inches='tight')
    
    # self.writer.add_summary(summary_str, counter)      
    plt.close(fig1)
    plt.close(fig3)

def dump_loss(norm_grad_list, g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list, args):
    
    foldername = 'results/' + str(args.scale) + '/optimal_gan/'
    foldername += args.teacher_name + '/lrg_' + str(args.lrg) + '_lrd_' + str(args.lrd) + '/'
    
    if args.teacher_name == 'rollout':
        foldername += str(args.rollout_steps) + '/' + args.rollout_method + '/' + str(args.rollout_rate) + '/'
    
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    filename = foldername + 'train-' + time.strftime("%m-%d-%H-%M-%S") + '.pkl'

    with open(filename, 'wb') as f: 
        pickle.dump([norm_grad_list, g_loss_list, d_loss_list, d_loss_real_list, args], f)
        print("dump loss to file ", filename)

class MovingAverage(object):
    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        from collections import deque

        self.size = size
        self.windowLen = 0
        self.windowSum = 0
        self.windowQue = deque()

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self.windowLen += 1
        self.windowQue.append( val )
        self.windowSum += val

        if self.windowLen > self.size:
            self.windowLen -= 1
            self.windowSum -= self.windowQue.popleft()

        return float(self.windowSum) / float(self.windowLen)


def plot_D_reward(prefix, sess, gan, it, scale=2.0, res=100):
    xres = res
    yres = res
    x = np.linspace(-scale, scale, xres)
    y = np.linspace(-scale, scale, yres)
    xx, yy = np.meshgrid(x,y)
    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]

    grid_logit = sess.run([gan.fake_logits], feed_dict={gan.fake_samples: xxyy})
    # reshape and plot image
    img = np.array(grid_logit).reshape((xres,yres))
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img)

    import h5py
    h5f = h5py.File(prefix + 'fig_%05d.h5' % it, 'w')
    h5f.create_dataset('reward_array', data=img)
    h5f.close()

    plt.show()

############################################################
####### Utils for MCTS #####################################
############################################################
def print_board(rewards, states_traj):
    plt.imshow(rewards)
    states_traj = np.array(states_traj)
    plt.plot(states_traj[:,0], states_traj[:,1])
    plt.show()

def reward_function(centers, scale, xlim, ylim, xres, yres):

    std = np.eye(2)/5
    mult_norm_array = [multivariate_normal(mean=center, cov=std) for center in centers]
    # import pdb; pdb.set_trace()

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x,y)
    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]

    zz = mult_norm_array[0].pdf(xxyy)
    for l in range(1,len(mult_norm_array)):
        zz += mult_norm_array[l].pdf(xxyy)

    # reshape and plot image
    img = zz.reshape((xres,yres))
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img)
    # ax.imshow(img, extent=[-scale,scale,-scale,scale]);
    plt.show()
    return img

def random_n_sphere(num, dim):
    normal_deviates = np.random.normal(size=(num,dim))
    radius = np.sqrt((normal_deviates**2).sum(axis=1))
    result = (normal_deviates.T / radius.clip(min=1e-10)).T

    # normal_deviates = torch.randn(size=(num,dim))
    # print(normal_deviates.shape)
    # radius = torch.sqrt(torch.sum((normal_deviates**2),dim=1))
    # print(radius.shape)
    # result = (normal_deviates.transpose / torch.clamp(radius,min=1e-10,max=1e10)).transpose
    # print(result.shape)

    return result

def draw_sample(model_batch,real_batch,scale,fname,color=None):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5,5)
    ax=fig.add_subplot(1,1,1)
    if real_batch is not None:
        ax.scatter(real_batch[:, 0], real_batch[:, 1], s=100, c='g')
    if color is None:
        color ='b'
    if model_batch is not None:
        ax.scatter(model_batch[:, 0], model_batch[:, 1], s=100, c=color)
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

def draw_landscape(grid_batch,grid_sigmoid,real_batch,scale,fname):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5,5)
    ax=fig.add_subplot(1,1,1)

    # Discriminator contour
    x_mesh = np.reshape(grid_batch[:,0],[int(np.sqrt(grid_batch.shape[0])),-1]).T
    y_mesh = np.reshape(grid_batch[:,1],[int(np.sqrt(grid_batch.shape[0])),-1]).T
    v_mesh = np.reshape(grid_sigmoid,[int(np.sqrt(grid_batch.shape[0])),-1]).T
    ax.contourf(x_mesh, y_mesh, v_mesh, 100, cmap='Greys', vmin = 0.0, vmax = 0.7)
    # ax.contourf(x_mesh, y_mesh, v_mesh, 100, cmap='Greys')

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

def draw_density(samps,scale,fname):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5,5)
    ax=fig.add_subplot(1,1,1)

    # Real samples
    import seaborn as sns
    sns.kdeplot(samps[:, 0], samps[:, 1], shade=True, cmap='Greys', gridsize=200, n_levels=100)

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

def pairwise_distance(real_batch,model_batch):
    dist_matrix = cdist(real_batch, model_batch)
    dist_eye = 10.0*np.identity(dist_matrix.shape[0])
    dist_min = np.min(dist_matrix+dist_eye,axis=0)
    return dist_min

# def metrics_distance(real_batch,model_batch,thres):
#     dist_min = pairwise_distance(real_batch,model_batch)
#     cnt_good = (dist_min < thres).sum()
#     cnt_all = dist_min.size
#     rate_good = cnt_good / cnt_all
#     mean_dist = np.mean(dist_min)
#     return mean_dist, rate_good

def metrics_distance(samples,centeroids,thres):
    samples = np.array(samples)
    centeroids = np.array(centeroids)
    n = np.size(samples,0)
    k = np.size(centeroids,0)
    distances = np.zeros((n,k))
    for i in range(k):
        distances[:,i] = np.linalg.norm(samples - centeroids[i], axis=1)
    dist_min = np.min(distances, axis=1)
    cnt_good = (dist_min < thres).sum()
    cnt_all = dist_min.size
    rate_good = cnt_good / cnt_all
    mean_dist = np.mean(dist_min)
    return mean_dist, rate_good

def freq_category(samples, centeroids, thres):
    k = np.size(centeroids,0)
    n = np.size(samples,0)
    distances = np.ones((n,k))
    for i in range(k):
        distances[:,i] = np.linalg.norm(samples - centeroids[i], axis=1)
    clusters = distances < thres 
    counts = np.sum(clusters,axis=0)
    sum_counts = np.sum(counts)
    if sum_counts > 0:
        freqs = counts / sum_counts
    else:
        freqs = np.ones(k) / k
    return freqs

def metrics_diversity(real_batch,model_batch,centeroids,thres):
    freq_real = freq_category(real_batch,centeroids,thres)
    freq_model = freq_category(model_batch,centeroids,thres)
    kl = kl_div(freq_real, freq_model)
    return kl

def kl_div(predictions, targets):
    """
    Input: predictions (k,1) ndarray
           targets (k,1) ndarray        
    Returns: scalar
    """
    targets = np.clip(targets, a_min = 1e-12, a_max = 1 - 1e-12)
    kl = stats.entropy(predictions, targets)
    return kl