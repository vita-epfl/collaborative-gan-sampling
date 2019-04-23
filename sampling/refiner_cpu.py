from __future__ import division
import os 
import numpy as np
np.random.seed(1234)

from utils import *
from policy import * 

class Refiner(object):
	"""docstring for Refiner"""
	def __init__(self, args):
		self.forward_steps = args.rollout_steps
		self.step_size = args.rollout_rate
		self.method = args.rollout_method
		self.args = args
		self.policy = PolicyAdaptive(self.step_size, self.method)
		self.log = False
		# self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))

	def set_env(self, gan, sess, data):
		self.sess = sess
		self.gan = gan
		self.data = data

	def visualize_rollout(self, x_traj, y_traj, fake_sigmoid, indices_optimal):
		xmedian, ymedian = np.median(x_traj), np.median(y_traj)
		grid_length = 0.5
		ngrids = 21
		grid_batch = np.zeros((ngrids * ngrids, 2))
		step = grid_length / (ngrids-1)
		for i in range(ngrids):
			for j in range(ngrids):
				grid_batch[i * ngrids + j, 0] = xmedian - grid_length / 2.0 + i * step
				grid_batch[i * ngrids + j, 1] = ymedian - grid_length / 2.0 + j * step
		grid_sigmoid = self.sess.run(self.gan.fake_sigmoid, feed_dict={self.gan.fake_samples: grid_batch})
		
		mean_sigmoid = np.mean(fake_sigmoid)
		plot_trajectories(self.ax, x_traj, y_traj, grid_batch, grid_sigmoid, mean_sigmoid, indices_optimal)

		prefix = 'figs/rollout/' + self.method + '/' + str(self.step_size) + '/'
		if not os.path.exists(prefix):
			os.makedirs(prefix)
		self.fig.savefig(prefix + 'rollout_%.4f.png' % mean_sigmoid, bbox_inches='tight')


	def normalize_gradient(self, optimal_grad, fake_sigmoid, fake_grad, real_sigmoid, real_grad):

		max_sigmoid = max(np.max(fake_sigmoid),np.max(real_sigmoid))
		min_sigmoid = min(np.min(fake_sigmoid),np.min(real_sigmoid))
		max_norm = max(np.max(np.linalg.norm(fake_grad, axis=1)), np.max(np.linalg.norm(real_grad, axis=1)))
		min_norm = min(np.min(np.linalg.norm(fake_grad, axis=1)), np.min(np.linalg.norm(real_grad, axis=1)))

		# clip
		# max_norm = min(1.0, max_norm)

		# rescale
		rescale_norm = (max_sigmoid - fake_sigmoid) / (max_sigmoid - min_sigmoid) * (max_norm - min_norm) + min_norm
		optimal_grad *= (rescale_norm / (np.expand_dims(np.linalg.norm(optimal_grad, axis=1)+1e-8, axis=1)) )

	def reweight_gradient(self, rollout_grad, default_grad, fake_sigmoid):

		default_norm = np.linalg.norm(default_grad, axis=1)
		rollout_norm = np.linalg.norm(rollout_grad, axis=1)

		weight_norm = 1-fake_sigmoid

		scale_norm = np.sum(default_norm) / np.sum(weight_norm[np.nonzero(rollout_norm)])

		rollout_norm = np.expand_dims(rollout_norm,axis=1)

		rollout_grad *= scale_norm*weight_norm/rollout_norm.clip(min=1e-12)
		
		rollout_norm = np.linalg.norm(rollout_grad, axis=1)
		# print("sum of rollout_norm = {:.8f}, default_norm = {:.8f}".format(np.sum(rollout_norm), np.sum(default_norm)))

	def revert_norm(self, rollout_grad, default_grad):
		default_norm = np.linalg.norm(default_grad, axis=1)
		rollout_norm = np.linalg.norm(rollout_grad, axis=1)
		rollout_grad *= np.expand_dims(default_norm/rollout_norm.clip(min=1e-12),axis=1)

	def manipulate_gradient(self, fake_batch, fake_sigmoid, fake_grad, task='2DGaussian'):

		if task == '2DGaussian':
			# real reference 
			real_batch = self.data.next_batch(fake_batch.shape[0])
			real_grad, real_sigmoid = self.sess.run([self.gan.fake_saliency, self.gan.fake_sigmoid], feed_dict={self.gan.fake_samples: real_batch})

			# variable  
			forward_batch = fake_batch.copy()
			forward_grad = fake_grad.copy()
			forward_loss = np.mean(real_sigmoid) - np.squeeze(fake_sigmoid)

			# optimal search 
			optimal_batch = forward_batch.copy()
			optimal_loss = forward_loss.copy()
			optimal_step = np.zeros_like(optimal_loss)

			# rollout trajectory 
			size_batch = len(fake_batch)
			x_traj = np.zeros((size_batch, self.forward_steps+1))
			y_traj = np.zeros((size_batch, self.forward_steps+1))
			l_traj = np.zeros((size_batch, self.forward_steps+1))

			x_traj[:,0] = fake_batch[:,0]
			y_traj[:,0] = fake_batch[:,1]
			l_traj[:,0] = forward_loss

			# recursive forward search 
			for i in range(self.forward_steps):

				# forward update 
				self.policy.apply_gradient(forward_batch, forward_grad, forward_loss)

				# compute current value and next grad 
				forward_sigmoid, forward_grad = self.sess.run([self.gan.fake_sigmoid, self.gan.fake_saliency], feed_dict={self.gan.fake_samples: forward_batch})
				
				if self.log:
					print("#", i+1, ": forward_grad = ", forward_grad[:3,0])
					reshape_forward_grad = np.reshape(forward_grad, (forward_grad.shape[0], -1))
					norm_forward_grad = np.mean(np.linalg.norm(reshape_forward_grad, axis=1))

				# states
				forward_loss = np.mean(real_sigmoid) - np.squeeze(forward_sigmoid)

				# comparison
				indices_update = (optimal_loss - forward_loss) > 0
				optimal_loss[indices_update] = forward_loss[indices_update]
				optimal_batch[indices_update,:] = forward_batch[indices_update,:]
				optimal_step[indices_update] = i+1

				# rollout trajectory 
				x_traj[:,i+1] = forward_batch[:,0]
				y_traj[:,i+1] = forward_batch[:,1]
				l_traj[:,i+1] = forward_loss

			# reset teacher
			self.policy.reset_moving_average()
			
			# trajectory review 
			indices_optimal = np.argmin(l_traj, axis=1)

			rollout_grad = np.zeros_like(fake_batch) 
			rollout_grad[:,0] = fake_batch[:,0] - x_traj[np.arange(len(indices_optimal)), indices_optimal]
			rollout_grad[:,1] = fake_batch[:,1] - y_traj[np.arange(len(indices_optimal)), indices_optimal]

			optimal_grad = (fake_batch - optimal_batch)

			# print("optimal_step: min = {:.1f}, max = {:.1f}，mean = {:.1f}, std = {:.1f}".format(np.min(optimal_step),np.max(optimal_step),np.mean(optimal_step),np.std(optimal_step)))
			# print("optimal_loss: min = {:.1f}, max = {:.1f}，mean = {:.1f}, std = {:.1f}".format(np.min(optimal_loss),np.max(optimal_loss),np.mean(optimal_loss),np.std(optimal_loss)))
			
			optimal_step = optimal_step.clip(min=1.0) 	# zero denominator 
			rollout_grad /= np.expand_dims(optimal_step, axis=1)
			optimal_grad /= np.expand_dims(optimal_step, axis=1)
			# assert np.max(np.abs(optimal_grad - rollout_grad)) < 1e-6

			# visualization
			# self.visualize_rollout(x_traj, y_traj, fake_sigmoid, indices_optimal)
		
		else:
			# real reference 
			real_batch = self.data[:self.args.batch_size]
			real_grad, real_sigmoid = self.sess.run([self.gan.saliency_map, self.gan.D], feed_dict={self.gan.inputs: real_batch})

			# compute current value and next grad
			forward_sigmoid, forward_grad = self.sess.run([self.gan.D, self.gan.saliency_map], feed_dict={self.gan.inputs: fake_batch})

			# variable
			forward_batch = fake_batch.copy()
			forward_grad = fake_grad.copy()
			forward_loss = np.mean(real_sigmoid) - np.squeeze(fake_sigmoid)
			
			# optimal search 
			optimal_batch = forward_batch.copy()
			optimal_loss = forward_loss.copy()
			optimal_step = np.zeros_like(optimal_loss)

			# recursive forward search 
			for i in range(self.forward_steps):
				
				# forward update 
				self.policy.apply_gradient(forward_batch, forward_grad, forward_loss)

				# compute current value and next grad
				forward_sigmoid, forward_grad = self.sess.run([self.gan.D, self.gan.saliency_map], feed_dict={self.gan.inputs: forward_batch})
				
				if self.log:
					# print("#", i + 1, ": forward_grad = {:.2e}".format(forward_grad[1,10,10,0]) )
					reshape_forward_grad = np.reshape(forward_grad, (forward_grad.shape[0], -1))
					norm_forward_grad = np.mean(np.linalg.norm(reshape_forward_grad, axis=1))

					reshape_fake_grad = np.reshape(fake_grad, (fake_grad.shape[0], -1))
					norm_fake_grad = np.mean(np.linalg.norm(reshape_fake_grad, axis=1))

				# states
				forward_loss = np.mean(real_sigmoid) - np.squeeze(forward_sigmoid)

				# comparison
				indices_update = (optimal_loss - forward_loss) > 0
				optimal_loss[indices_update] = forward_loss[indices_update]
				optimal_batch[indices_update,:,:,:] = forward_batch[indices_update,:,:,:]
				optimal_step[indices_update] = i+1

			# reset teacher
			self.policy.reset_moving_average()

			optimal_grad = (fake_batch - optimal_batch)
			optimal_step = optimal_step.clip(min=1.0)
			optimal_grad /= optimal_step[:,None,None,None]

			# TODO: precision 32 vs 64
			# assert np.max(np.abs(fake_grad - optimal_grad)) < 1e-8

		# TODO: magnitude verification 
		# print("fake_grad = ", fake_grad[:3,10,10,0])
		# print("forward_grad = ", forward_grad[:3,10,10,0])
		# print("optimal_grad = ", optimal_grad[:3,10,10,0])

		if self.log:
			reshape_fake_grad = np.reshape(fake_grad, (fake_grad.shape[0], -1))
			norm_fake_grad = np.mean(np.linalg.norm(reshape_fake_grad, axis=1))
			reshape_optimal_grad = np.reshape(optimal_grad, (optimal_grad.shape[0], -1))
			norm_optimal_grad = np.mean(np.linalg.norm(reshape_optimal_grad, axis=1))
			print("grad norm: fake = {:.2e}, forward = {:.2e}, rollout = {:.2e}, optimal_step = {:d}".format(norm_fake_grad, norm_forward_grad, norm_optimal_grad, optimal_step) )

		# print("optimal_step: variance = {:.1f}, median = {:.1f}, mean = {:.1f}".format(np.var(optimal_step),np.median(optimal_step),np.mean(optimal_step)))
		# self.normalize_gradient(optimal_grad, fake_sigmoid, fake_grad, real_sigmoid, real_grad)
		# self.reweight_gradient(optimal_grad, fake_grad, fake_sigmoid)
		# self.revert_norm(optimal_grad, fake_grad)

		return optimal_grad, optimal_batch

	def manipute_latent(self, z_batch, fake_sigmoid, z_grad, task='2DGaussian'):

		# real reference 
		real_batch = self.data.next_batch(z_batch.shape[0])
		real_grad, real_sigmoid = self.sess.run([self.gan.fake_saliency, self.gan.fake_sigmoid], feed_dict={self.gan.fake_samples: real_batch})

		# variable  
		forward_batch = z_batch.copy()
		forward_grad = z_grad.copy()
		forward_loss = np.mean(real_sigmoid) - np.squeeze(fake_sigmoid)

		# optimal search 
		optimal_batch = forward_batch.copy()
		optimal_loss = forward_loss.copy()
		optimal_step = np.zeros_like(optimal_loss)

		# recursive forward search 
		for i in range(self.forward_steps):

			# forward update 
			self.policy.apply_gradient(z_batch, z_grad, forward_loss)
			# z_batch -= self.step_size * z_grad
			
			# compute current value and next grad 
			forward_batch, forward_sigmoid, z_grad = self.sess.run([self.gan.generates, self.gan.fake_sigmoid, self.gan.grad_z], feed_dict={self.gan.z: z_batch})

			# states
			forward_loss = np.mean(real_sigmoid) - np.squeeze(forward_sigmoid)

			# comparison
			indices_update = (optimal_loss - forward_loss) > 0
			optimal_loss[indices_update] = forward_loss[indices_update]
			optimal_batch[indices_update,:] = forward_batch[indices_update,:]
			optimal_step[indices_update] = i+1

		# reset teacher
		self.policy.reset_moving_average()

		print("optimal_step: ", np.mean(optimal_step), np.std(optimal_step))

		return optimal_batch
