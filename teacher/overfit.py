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

data = ToyDataset(distr="Imbal-8Gaussians", scale=10.0)
batch = data.next_batch(80)
print(batch.size)
draw_sample(None,batch,scale=10.0,fname='temp.png')


# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
nbins=300
xi, yi = np.mgrid[-10.0:10.0:nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
 
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# Make the plot
plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
plt.show()


# Make the plot
plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
plt.show()
