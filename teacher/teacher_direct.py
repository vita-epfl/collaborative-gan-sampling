from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

class TeacherDirect(object):
    """docstring for TeacherScalized"""
    def __init__(self, scale=2.0, centers=None):
        from sklearn.preprocessing import normalize
        self.scale = scale
        self.centers = [
                        (1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1),
                        (1. / np.sqrt(2), 1. / np.sqrt(2)),
                        (1. / np.sqrt(2), -1. / np.sqrt(2)),
                        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                        (-1. / np.sqrt(2), -1. / np.sqrt(2))
                        ]
        self.centers = [((scale / 1.414) * x, (scale/1.414) * y) for x, y in self.centers]
        if centers is not None:
            self.centers = centers
        self.normalize = normalize

		
    def set_env(self, sess,gan):
        self.sess = sess
        self.gan = gan

    def manipulate_gradient(self, fake_batch, fake_logit, fake_grad):
        f = fake_batch
        r = np.array(self.centers)
        distance = np.sqrt(np.sum(np.square(np.subtract(r, np.expand_dims(f,1))), axis=2))
        # print(distance.shape)
        ind = np.argmin(distance, axis=1)
        goals = np.take(r, indices=ind, axis=0)
        grads = goals - f
        ##To STILL Think this Over
        # new_grad = -grads
        # print(goals.shape)
        # print(f.shape)
        new_grad = -self.normalize(grads, norm='l2', axis=1)
        # print(fake_grad.shape)
        # print(new_grad.shape)

        return new_grad
