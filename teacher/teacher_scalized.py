from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from utils import MovingAverage

class TeacherScalized(object):
	"""docstring for TeacherScalized"""
	def __init__(self, args):
		# moving average features
		self.scale_norm_ma = 1.0
		self.fake_logit_ma = 0.0
		# parameters
		self.thres_norm_ratio = 5e3
		self.thres_logit = 1e-2
		self.factor_norm_scale = 1.1
		# operator
		self.ma_operator = MovingAverage(10)
		
	def set_env(self, sess,gan):
		self.sess = sess
		self.gan = gan

	def manipulate_gradient(self, fake_batch, fake_logit, fake_grad):

		new_grad = fake_grad 

		fake_logit_mean = np.mean(fake_logit)
		print("fake_logit = {:.6f}, ma = {:.6f}, scalar = {:.2f}".format( fake_logit_mean, self.fake_logit_ma, self.scale_norm_ma) )
		if fake_logit_mean - self.fake_logit_ma < self.thres_logit * self.fake_logit_ma:
			self.scale_norm_ma *= self.factor_norm_scale
			new_grad *= self.scale_norm_ma
		else:
			self.scale_norm_ma /= self.factor_norm_scale

		self.scale_norm_ma = max(min(self.scale_norm_ma, self.thres_norm_ratio), 1.0)
		self.fake_logit_ma = self.ma_operator.next(fake_logit_mean)

		return new_grad
