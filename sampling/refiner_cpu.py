from __future__ import division
import numpy as np

from policy import PolicyAdaptive

class Refiner():
    """docstring for Refiner"""
    def __init__(self, args):
        self.forward_steps = args.rollout_steps
        self.step_size = args.rollout_rate
        self.method = args.rollout_method
        self.policy = PolicyAdaptive(self.step_size, self.method)

    def set_env(self, gan, sess, data):
        self.sess = sess
        self.gan = gan
        self.data = data

    def manipulate_sample(self, fake_batch, mode='deterministic'):

        # real reference
        real_batch = self.data.next_batch(fake_batch.shape[0])
        real_grad, real_sigmoid = self.sess.run([self.gan.fake_saliency, self.gan.fake_sigmoid], feed_dict={self.gan.fake_samples: real_batch})

        # variable
        forward_batch = fake_batch.copy()
        forward_sigmoid, forward_grad = self.sess.run([self.gan.fake_sigmoid, self.gan.fake_saliency], feed_dict={self.gan.fake_samples: forward_batch})
        forward_loss = np.mean(real_sigmoid) - np.squeeze(forward_sigmoid)

        # optimal search
        optimal_batch = forward_batch.copy()
        optimal_loss = forward_loss.copy()
        optimal_step = np.zeros_like(optimal_loss)

        # rollout trajectory
        size_batch = len(fake_batch)
        x_traj = np.zeros((size_batch, self.forward_steps+1))
        y_traj = np.zeros((size_batch, self.forward_steps+1))
        l_traj = np.zeros((size_batch, self.forward_steps+1))

        x_traj[:, 0] = fake_batch[:, 0]
        y_traj[:, 0] = fake_batch[:, 1]
        l_traj[:, 0] = forward_loss

        # recursive forward search
        for i in range(self.forward_steps):

            # forward update
            self.policy.apply_gradient(forward_batch, forward_grad, forward_loss)

            # compute current value and next grad
            forward_sigmoid, forward_grad = self.sess.run([self.gan.fake_sigmoid, self.gan.fake_saliency], feed_dict={self.gan.fake_samples: forward_batch})

            # states
            forward_loss = np.mean(real_sigmoid) - np.squeeze(forward_sigmoid)

            # comparison
            indices_update = (optimal_loss - forward_loss) > 0
            optimal_loss[indices_update] = forward_loss[indices_update]
            optimal_batch[indices_update, :] = forward_batch[indices_update, :]
            optimal_step[indices_update] = i+1

            # rollout trajectory
            x_traj[:, i+1] = forward_batch[:, 0]
            y_traj[:, i+1] = forward_batch[:, 1]
            l_traj[:, i+1] = forward_loss

        # reset teacher
        self.policy.reset_moving_average()

        if mode == 'probabilistic':
            indices_batch = np.random.randint(self.forward_steps+1, size=size_batch)
            x_batch = x_traj[np.indices(indices_batch.shape)[0], indices_batch]
            y_batch = y_traj[np.indices(indices_batch.shape)[0], indices_batch]
            refined_batch = np.array([x_batch, y_batch]).transpose()
            return refined_batch

        if mode == 'deterministic':
            return optimal_batch

        raise NotImplementedError
