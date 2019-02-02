from __future__ import division
import os
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys

from utils import draw_sample 

np.random.seed(1234)

sys.path.append('.')
sys.path.append(os.path.join('..', '2DGaussians'))
from Datasets import * 

def parse_args():
    desc = "Figures"
    parser = argparse.ArgumentParser(description=desc)	
    parser.add_argument('--dataset', type=str, default='8Gaussians',
                        help='dataset to use: 8Gaussians | 25Gaussians | swissroll | mnist')
    parser.add_argument('--scale', type=float, default=10., help='data scaling')
    parser.add_argument('--batch_size', type=int, default=400, help='batch size')    
    return parser.parse_args()

def draw_real_8G():
	args = parse_args()
	data = ToyDataset(distr=args.dataset, scale=args.scale)	
	real_batch = data.next_batch(args.batch_size)
	draw_sample(real_batch,args.scale*0.8,'8G_real.pdf')

def draw_real_25G():
	args = parse_args()
	args.dataset = '25Gaussians'
	data = ToyDataset(distr=args.dataset, scale=args.scale)	
	real_batch = data.next_batch(args.batch_size)
	draw_sample(real_batch,args.scale*1.6,'25G_real.pdf')


def draw_real_sr():
	args = parse_args()
	args.dataset = 'swissroll'
	data = ToyDataset(distr=args.dataset, scale=args.scale)	
	real_batch = data.next_batch(args.batch_size)
	draw_sample(real_batch,args.scale*1.2,'sr_real.pdf')


if __name__ == '__main__':
	# draw_real_8G()
    draw_real_25G()
	# draw_real_sr()