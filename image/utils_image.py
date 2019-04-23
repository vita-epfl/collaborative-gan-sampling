"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import copy
import random
import pprint
import scipy.misc
import numpy as np
import pickle
import os 
import shutil
import time
from time import gmtime, strftime
from six.moves import xrange
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from operator import itemgetter
from scipy.stats import multivariate_normal

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def show_images(images, size, ax=None):
  return imgshow(inverse_transform(images), size, ax)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def imgshow(images, size, ax=None):
  image = np.squeeze(merge(images, size))
  # print(image.shape)
  if ax is None:
    plt.imshow(image)
    plt.draw()
    # plt.pause(1e-6)
    # plt.show(block = False)
  else:
    ax.clear()
    ax.imshow(image)
  return image

def plot_image(image, ax, it, desc):
    ax.clear()

    ax.imshow(image, vmin=-1.0, vmax=1.0)

    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    ax.set_title(desc + " Iter #{:d}".format(it))

    # plt.legend()
    plt.draw()
    # plt.show()

def plot_figs(input_batch, optimal_batch, epoch, itr, config, show_horizontal=True):

    #IF showing images as grid (not horizontal), num_plot must be a square number!!
    if config.dataset == 'mnist':  
      num_plot = 12
    else:
      num_plot = 12

    ##Fig 1
    size_square = image_manifold_size(num_plot)
    size_horizontal = (1, size_square[0]*size_square[1])
    
    if not show_horizontal:
      fig1, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(9, 9))
      size = size_square
    else:
      fig1, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(15, 3))
      size = size_horizontal
  
    thres = -1.1
    opt_batch = optimal_batch[:num_plot]
    noisy = opt_batch < thres  
    opt_batch[noisy] = -1

    inp_batch = input_batch[:num_plot]
    noisy = inp_batch < thres  
    inp_batch[noisy] = -1

    image_default = show_images(inp_batch, size, ax1)
    # print(image_default.shape)
    # ax1.set_title("Input Image")
    ax1.axis('off')
    ax1_extent = ax1.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    image_optimal = show_images(opt_batch, size, ax2)
    # ax2.set_title("Optimal Image")
    ax2.axis('off')
    ax2_extent = ax2.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())

    image_difference = show_images((opt_batch - inp_batch)+0.5, size, ax3)
    # ax3.set_title("Diff Image")
    ax3.axis('off')
    ax3_extent = ax3.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())

    plt.figure(fig1.number)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.draw()
    # plt.pause(1e-3)
    # plt.show()

    # print(config.dataset, config.dataset, config.mode, config.load_epoch, config.rollout_steps, config.rollout_rate, config.rollout_method, config.collab_layer)
    if config.save_figs:
      save_dir = './figs/{}/viz_{}_{}_load_epoch_{}_rs_{:03d}_rr_{:.2f}_opt_{}_layer_{}'.format(config.dataset, config.dataset, config.mode, config.load_epoch, config.rollout_steps, config.rollout_rate, config.rollout_method, config.collab_layer)
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)

      print("Saving Figs")
      fig1.savefig(save_dir + '/epoch_{:02d}_{:04d}.png'.format(epoch, itr), bbox_inches='tight')
      fig1.savefig(save_dir + '/epoch_{:02d}_{:04d}_default.png'.format(epoch, itr), bbox_inches=ax1_extent)
      fig1.savefig(save_dir + '/epoch_{:02d}_{:04d}_optimal.png'.format(epoch, itr), bbox_inches=ax2_extent)
      fig1.savefig(save_dir + '/epoch_{:02d}_{:04d}_diff.png'.format(epoch, itr), bbox_inches=ax3_extent)
    
    plt.close(fig1)
    
    # print("Saving Individual Figs")
    # plot_single_fig(image_default, num_plot, config.dataset) ## Options Default, Optimal, Difference
    

def plot_single_fig(input_batch, filename, foldername='./indi_figs', vmin=None, vmax=None):

    if not os.path.exists(foldername):
      os.makedirs(foldername)
      print("Create folder: ", foldername)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    size = (1,1)

    for idx in xrange(np.size(input_batch,0)):
      inp_batch = np.expand_dims(input_batch[idx],axis=0)
      image = show_images(inp_batch, size, ax)
      extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
      if vmin is not None or vmax:
        plt.imshow(image,vmin=vmin,vmax=vmax)
      else:
        plt.imshow(image)
      plt.axis('off')
      plt.savefig(foldername + '/' + filename + "_" + str(idx) + '.png', bbox_inches=extent, pad_inches=0)
      ax.clear()
    plt.close()

def plot_pairs(input_batch, optimal_batch, epoch, itr, config, show_horizontal=True):

    # diff_sigmoid = optimal_sigmoid - input_sigmoid
    # order_sigmoid = np.argsort(diff_sigmoid[::-1])
    num_plot = 1

    ##Fig 1
    size_square = image_manifold_size(num_plot)
    size_horizontal = (1, size_square[0]*size_square[1])
    
    if not show_horizontal:
      fig1, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(9, 9))
      size = size_square
    else:
      fig1, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(1, 3))
      size = size_horizontal

    batch_size = np.size(optimal_batch, axis=0)

    for idx in xrange(batch_size):
      thres = -1.1
      opt_batch = np.expand_dims(optimal_batch[idx],axis=0)
      noisy = opt_batch < thres  
      if np.sum(noisy) > 0:
        print("warning: np.sum(noisy)",np.sum(noisy))
      opt_batch[noisy] = -1

      inp_batch = np.expand_dims(input_batch[idx],axis=0)
      noisy = inp_batch < thres  
      if np.sum(noisy) > 0:
        print("warning: np.sum(noisy)",np.sum(noisy))
      inp_batch[noisy] = -1

      show_images(inp_batch, size, ax1)
      # ax1.set_title("Input Image")
      ax1.axis('off')
      ax1_extent = ax1.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
      
      show_images(opt_batch, size, ax2)
      # ax2.set_title("Optimal Image")
      ax2.axis('off')
      ax2_extent = ax2.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())

      show_images((opt_batch - inp_batch)+0.5, size, ax3)
      # ax3.set_title("Diff Image")  
      ax3.axis('off')
      ax3_extent = ax3.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())

      plt.figure(fig1.number)
      plt.subplots_adjust(wspace=0, hspace=0)
      plt.draw()
      # plt.pause(1e-3)
      # plt.show()

      if config.save_figs:
        save_dir = './figs/{}/viz_{}_{}_load_epoch_{}_rs_{:03d}_rr_{:.2f}_opt_{}_layer_{}'.format(config.dataset, config.dataset, config.mode, config.load_epoch, config.rollout_steps, config.rollout_rate, config.rollout_method, config.collab_layer)
        if not os.path.exists(save_dir):
          os.makedirs(save_dir)

        # if diff_sigmoid[idx] < 0.1:
        fig1.savefig(save_dir + '/epoch_{:02d}_{:04d}.png'.format(epoch, itr), bbox_inches='tight')
        # fig1.savefig(save_dir + '/epoch_{:02d}_{:04d}_{:.4f}_default.png'.format(epoch, itr), bbox_inches=ax1_extent)
        # fig1.savefig(save_dir + '/epoch_{:02d}_{:04d}_{:.4f}_optimal.png'.format(epoch, itr), bbox_inches=ax2_extent)
        # fig1.savefig(save_dir + '/epoch_{:02d}_{:04d}_{:.4f}_diff.png'.format(epoch, itr), bbox_inches=ax3_extent)

    plt.close(fig1)

def plot_single_fig(input_batch, filename, foldername='./indi_figs', vmin=None, vmax=None):

    if not os.path.exists(foldername):
      os.makedirs(foldername)
      print("Create folder: ", foldername)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    size = (1,1)

    for idx in xrange(np.size(input_batch,0)):
      inp_batch = np.expand_dims(input_batch[idx],axis=0)
      image = show_images(inp_batch, size, ax)
      extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
      if vmin is not None or vmax:
        plt.imshow(image,vmin=vmin,vmax=vmax)
      else:
        plt.imshow(image)
      plt.axis('off')
      plt.savefig(foldername + '/' + filename + "_" + str(idx) + '.png', bbox_inches=extent, pad_inches=0)
      ax.clear()
    plt.close()


def plot_fig_samples(samples):
  fig2, ax_2 = plt.subplots(1, 1, figsize=(5, 5))
  plt.figure(fig2.number)
  ax_2.clear()
  show_images(samples, image_manifold_size(samples.shape[0]))
  plt.close(fig2)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  if option == 0:
    print("CHOSE OPTION 0")
    z_sample = np.random.uniform(-1.0, 1.0, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    show_images(samples, [image_frame_dim, image_frame_dim])
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

        # samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, dcgan.z_dim - 1) for _ in xrange(dcgan.z_dim)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w


############################################################
####### GAN Viz ############################################
############################################################
def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def plot_loss(prefix, norm_grad_list, g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list, refiner_name = None, rollout_method = None, rollout_steps = None, rollout_rate = None):
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
    
    title = refiner_name
    if refiner_name == 'rollout':
        title += ': ' + rollout_method + ', step={:d}, rate={:.1e}'.format(rollout_steps,rollout_rate)
    ax.set_title(title)

    plt.xlabel('Step')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(prefix + 'metrics.png')

def plot_samples(prefix, ax, scale, real_batch, grid_batch, grid_grad, grid_sigmoid, it, loss, norm_grad, fake_batch=None, refiner_batch=None, perturbed_batch=None, xmin=None, xmax=None, ymin=None, ymax=None, grid_grad_refiner=None, title=None):
    # print("Iter: %d, loss: %.8f, norm_grad = %.8f" % (it, loss, norm_grad))
    
    ax.clear()

    # Discriminator contour
    if grid_batch is not None:
        x_mesh = np.reshape(grid_batch[:,0],[int(np.sqrt(grid_batch.shape[0])),-1]).T
        y_mesh = np.reshape(grid_batch[:,1],[int(np.sqrt(grid_batch.shape[0])),-1]).T
        v_mesh = np.reshape(grid_sigmoid,[int(np.sqrt(grid_batch.shape[0])),-1]).T
        ax.contourf(x_mesh, y_mesh, v_mesh, 50, cmap='Greys', vmin = 0.2, vmax = 0.8)

    # norm_grad_mean = np.mean(np.linalg.norm(grid_grad_refiner, axis=1))
    # norm_grad_mean = (np.mean(np.linalg.norm(grid_grad, axis=1)) + np.mean(np.linalg.norm(grid_grad_refiner, axis=1))) / 2
    # , scale=norm_grad_mean*10

        ax.quiver(grid_batch[:, 0], grid_batch[:, 1], -grid_grad[:, 0], -grid_grad[:, 1], color='b')
        if grid_grad_refiner is not None:
            ax.quiver(grid_batch[:, 0], grid_batch[:, 1], -grid_grad_refiner[:, 0], -grid_grad_refiner[:, 1], color='g')

    # sample distribution
    if perturbed_batch is not None:
        ax.scatter(perturbed_batch[:, 0], perturbed_batch[:, 1], s=10, c='c', label='perturb')        
    if real_batch is not None:
        ax.scatter(real_batch[:, 0], real_batch[:, 1], s=10, c='y', label='real')
    if refiner_batch is not None:
        ax.scatter(refiner_batch[:, 0], refiner_batch[:, 1], s=10, c='g', label='target')
    if fake_batch is not None:
        ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=10, c='r', label='fake')

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

def plot_norm_distribution(ax, loss, grad_default, grad_refiner, refiner_name, rollout_method, rollout_steps, rollout_rate):

    UB_NORM = 1.0
    LB_NORM = 1e-5    

    ax.clear()

    if grad_default.ndim > 2:
        grad_default = np.reshape(grad_default,(grad_default.shape[0],-1))

    norm_grad_default = np.linalg.norm(grad_default, axis=1)

    ax.semilogy(np.squeeze(loss), norm_grad_default.clip(min=LB_NORM,max=UB_NORM), 'bo', label='default')

    if grad_refiner is not None:
        if grad_refiner.ndim > 2:
            grad_refiner = np.reshape(grad_refiner,(grad_refiner.shape[0],-1))
        norm_grad_refiner = np.linalg.norm(grad_refiner, axis=1)
        ax.semilogy(np.squeeze(loss), norm_grad_refiner.clip(min=LB_NORM,max=UB_NORM), 'gx', label='refiner')

    ax.set_xlabel("loss\' = $1 - D(G(z))$")
    ax.set_ylabel("grad norm")
    ax.set_xlim((0.0,1.0))
    ax.set_ylim((LB_NORM, UB_NORM))

    title = refiner_name
    if refiner_name == 'rollout':
        title += ': ' + rollout_method + ', step={:d}, rate={:.1e}'.format(rollout_steps,rollout_rate)
    ax.set_title(title)

    plt.legend()

    plt.draw()
    plt.pause(1e-6)
    # plt.show(block = False)


def plot_norm_histogram(ax, loss, grad_default, grad_refiner, refiner_name, rollout_method, rollout_steps, rollout_rate):

    ax.clear()

    # the histogram of the data
    n, bins, patches = plt.hist(loss, bins=50, density=1, facecolor='green')

    ax.set_xlabel("$\Delta S$")
    ax.set_ylabel("Probability")
    ax.set_xlim((0.0,1.0))
    ax.set_ylim((0.0,20.0))

    ax.grid(True)

    title = refiner_name
    if refiner_name == 'rollout':
        title += ': ' + rollout_method + ', step={:d}, rate={:.1e}'.format(rollout_steps,rollout_rate)
    ax.set_title(title)

    plt.legend()

    plt.draw()
    plt.pause(1e-6)
    # plt.show(block = False)


def dump_loss(norm_grad_list, g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list, args):
    
    foldername = 'results/' + str(args.scale) + '/optimal_gan/'
    foldername += args.refiner_name + '/lrg_' + str(args.lrg) + '_lrd_' + str(args.lrd) + '/'
    
    if args.refiner_name == 'rollout':
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

    # plt.show(block = False)

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
    # plt.show()
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

def make_folders(FLAGS, mnist=True, dcgan=False):
  if not dcgan:
    ckpt = 'checkpoints'
    samp = 'samples'
    sc = 'scores'
    fi = 'figs'
  else:
    ckpt = 'dc_checkpoints'
    samp = 'dc_samples'
    sc = 'dc_scores'
    fi = 'dc_figs'   
    
  if mnist:
    if not os.path.exists(ckpt):
      os.makedirs(ckpt)

    if not os.path.exists(ckpt + '/mnist'):
      os.makedirs(ckpt + '/mnist')

    if not os.path.exists(ckpt + '/cifar10'):
      os.makedirs(ckpt + '/cifar10')

    if not os.path.exists(samp):
      os.makedirs(samp)

    if not os.path.exists(samp + '/mnist'):
      os.makedirs(samp + '/mnist')

    if not os.path.exists(samp + '/cifar10'):
      os.makedirs(samp + '/cifar10')

    if not os.path.exists(sc):
      os.makedirs(sc)

    if not os.path.exists(sc +'/mnist'):
      os.makedirs(sc +'/mnist')

    if not os.path.exists(sc +'/cifar10'):
      os.makedirs(sc +'/cifar10')

    FLAGS.checkpoint_dir = ckpt + '/{}/epoch_{}_refiner_{}_rollout_method_{}_rollout_steps_{}_rollout_rate_{:06.5f}'.format(
                            FLAGS.dataset, FLAGS.epoch, FLAGS.refiner_name, FLAGS.rollout_method, FLAGS.rollout_steps, FLAGS.rollout_rate)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)

    if FLAGS.refiner_name == 'default':
      FLAGS.sample_dir =  samp + '/{}/epoch_{}_refiner_{}_LRD_{}_LRG_{}_Dit_{}_Git_{}'.format(
                          FLAGS.dataset, FLAGS.epoch, FLAGS.refiner_name, FLAGS.D_LR, FLAGS.G_LR, FLAGS.D_it, FLAGS.G_it)

    else: 
      FLAGS.sample_dir =  samp + '/{}/epoch_{}_refiner_{}_rollout_method_{}_rollout_steps_{}_rollout_rate_{:06.5f}'.format(
                          FLAGS.dataset, FLAGS.epoch, FLAGS.refiner_name, FLAGS.rollout_method, FLAGS.rollout_steps, FLAGS.rollout_rate)

    if not os.path.exists(FLAGS.sample_dir):
      os.makedirs(FLAGS.sample_dir)

    print('FLAGS Ckpt Directory: ', FLAGS.checkpoint_dir)
    print('FLAGS Sample Directory: ', FLAGS.sample_dir)


  if not os.path.exists(fi):
    os.makedirs(fi)

  if not os.path.exists(fi + '/mnist'):
    os.makedirs(fi + '/mnist')

  if not os.path.exists(fi + '/cifar'):
    os.makedirs(fi + '/cifar')

def count_frequency(y):
  unique, counts = np.unique(y, return_counts=True)
  percents = counts / np.sum(counts)
  freq = dict(zip(unique, percents))
  return freq

def create_imbalance(X, y, targets, ratio):
    X_imbal = X
    y_imbal = y
    for target in targets:
        idxs = np.where(y_imbal==target)
        first_imbal = int(np.size(idxs)*(1-ratio))-1
        idxs_remove = idxs[0][:first_imbal]
        X_imbal = np.delete(X_imbal, idxs_remove, 0)
        y_imbal = np.delete(y_imbal, idxs_remove)
    return X_imbal, y_imbal
    