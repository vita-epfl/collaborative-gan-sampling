import numpy as np
import copy
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

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

