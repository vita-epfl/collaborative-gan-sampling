from __future__ import division
import numpy as np

class IndependenceSampler():
    def __init__(self, T=5, B=0):
        self.d_curr = None
        self.cnt_chain = 1
        self.thin_period = T
        self.burn_in = B

    def set_score_curr(self, d_curr):
        '''
        burn-in
        '''
        self.d_curr = d_curr

    def sampling(self, samples, sigmoids):
        '''
        mh
        '''
        assert samples.shape[0] == sigmoids.shape[0]
        assert np.min(sigmoids) >= 0.0
        assert np.max(sigmoids) <= 1.0
        good_samples = []
        curr_sample = None
        cnt_good = 0
        for i in range(samples.shape[0]):
            # move
            if self.next(sigmoids[i]):
                cnt_good = cnt_good + 1
                if cnt_good > self.burn_in:
                    curr_sample = samples[i]
            # accept
            if curr_sample is not None:
                if self.cnt_chain > self.thin_period:
                    good_samples.append(curr_sample)
                    self.cnt_chain = 1
                else:
                    self.cnt_chain = self.cnt_chain + 1

        return np.asarray(good_samples, dtype=np.float32)

    def next(self, d_next):
        '''
        @ param: x_next -> new sample
        @ param: d_next -> score from discriminator
        '''
        if self.d_curr is not None:
            alpha = min(1.0, d_next * (1.0 - self.d_curr) / (self.d_curr * (1.0 - d_next)))
            if np.random.uniform(0, 1) > alpha:
                return False
        self.d_curr = d_next
        return True
