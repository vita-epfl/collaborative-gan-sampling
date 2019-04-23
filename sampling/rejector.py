from __future__ import division
import os
import numpy as np
np.random.seed(1234)

from scipy.special import logit
from scipy.special import expit as logistic

class Rejector(object):
	def __init__(self, args):
		self.args = args
		self.D_tilde_M = 0.0

	def set_score_max(self, score_max):
		max_burnin_score = np.clip(score_max.astype(np.float), 1e-14, 1 - 1e-14)
		log_M = logit(max_burnin_score)
		self.D_tilde_M = np.maximum(self.D_tilde_M, log_M)
		# print("update D_tilde_M", self.D_tilde_M)

	def sampling(self, samples, sigmoids, epsilon=1e-8, shift_percent=95.0, rank=None):

		sigmoids = np.clip(sigmoids.astype(np.float), 1e-14, 1 - 1e-14)

	    # Update upper bound
		D_tilde = logit(sigmoids)
		self.D_tilde_M = np.maximum(self.D_tilde_M, np.amax(D_tilde))

	    # Compute probability
		D_delta = D_tilde - self.D_tilde_M 
		F = D_delta - np.log(1 - np.exp(D_delta - epsilon))
		if shift_percent is not None:
			gamma = np.percentile(F, shift_percent)
			# print("gamma", gamma)
			F = F - gamma
		P = np.squeeze(logistic(F))

		# Filter out samples
		# accept = np.random.rand(len(D_delta)) < P
		# good_samples = samples[accept]
		# print("[!] total: {:d}, accept: {:d}, percent: {:.2f}".format(len(D_delta), np.sum(accept), np.sum(accept)/len(D_delta) ))

		if rank is not None:
			order = np.argsort(P)[::-1]
			accept =  order[:int(rank*len(D_delta))]
			good_samples = samples[accept,:]
			print("[!] total: {:d}, accept: {:d}, percent: {:.2f}".format(len(D_delta), np.size(accept,0), np.size(accept,0)/len(D_delta) ))
		else:
			accept = np.random.rand(len(D_delta)) < P
			good_samples = samples[accept]
			print("[!] total: {:d}, accept: {:d}, percent: {:.2f}".format(len(D_delta), np.sum(accept), np.sum(accept)/len(D_delta) ))

		return good_samples
