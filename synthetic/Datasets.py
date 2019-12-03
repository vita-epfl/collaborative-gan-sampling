import numpy as np

class NoiseDataset:
    def __init__(self, distr='Gaussian', dim=2, var=1):
        self.distr = distr
        self.dim = dim
        self.var = var

    def next_batch(self, batch_size=64):
        if self.distr == 'Gaussian':
            return np.random.randn(batch_size, self.dim)
        else:
            raise NotImplementedError


class ToyDataset:
    def __init__(self, distr='8Gaussians', scale=2, ratio=0.5):
        self.distr = distr
        self.scale = scale
        self.ratio = ratio

        if self.distr == '8Gaussians' or self.distr == 'Imbal-8Gaussians':
            self.std = 0.02
            self.centers = [
                (1, 0),
                (1. / np.sqrt(2), 1. / np.sqrt(2)),
                (1. / np.sqrt(2), -1. / np.sqrt(2)),
                (-1, 0),
                (0, 1),
                (0, -1),
                (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                (-1. / np.sqrt(2), -1. / np.sqrt(2))
            ]
            self.centeroids = np.array(self.centers) * self.scale / 1.414
            self.std *= self.scale / 1.414

        if self.distr == '25Gaussians':
            self.std = 0.05
            self.centers = []
            for x in range(-2, 3):
                for y in range(-2, 3):
                    self.centers.append((x, y))
            self.centeroids = np.array(self.centers) * self.scale

    def next_batch(self, batch_size=64):
        if self.distr == '8Gaussians':
            num_repeat = int(batch_size/8)
            batch = np.repeat(self.centeroids, num_repeat, axis=0)
            num_random = batch_size - num_repeat * 8
            if num_random > 0:
                newrow = self.centeroids[np.random.randint(8, size=num_random), :]
                batch = np.concatenate([batch, newrow])
            noise = np.random.normal(0.0, self.std, size=(batch_size, 2))
            return batch + noise

        if self.distr == 'Imbal-8Gaussians':
            num_rep_maj = int(batch_size*self.ratio/2)
            assert num_rep_maj > 0
            majority = np.repeat(self.centeroids[:2, :], num_rep_maj, axis=0)
            num_rep_min = int((batch_size-num_rep_maj*2)/6)
            assert num_rep_min > 0
            minority = np.repeat(self.centeroids[2:, :], num_rep_min, axis=0)
            num_random = batch_size-num_rep_maj*2-num_rep_min*6
            if num_random > 0:
                newrow = self.centeroids[np.random.randint(8, size=num_random), :]
                batch = np.concatenate([majority, minority, newrow])
            else:
                batch = np.concatenate([majority, minority])
            noise = np.random.normal(0.0, self.std, size=(batch_size, 2))
            return batch + noise

        if self.distr == '25Gaussians':
            num_repeat = int(batch_size/25)
            batch = np.repeat(self.centeroids, num_repeat, axis=0)
            num_random = batch_size - num_repeat * 25
            if num_random > 0:
                newrow = self.centeroids[np.random.randint(25, size=num_random), :]
                batch = np.concatenate([batch, newrow])
            noise = np.random.normal(0.0, self.std, size=(batch_size, 2))
            return batch + noise

        # if self.distr == 'swissroll':
        #     data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=0.25)[0]
        #     data = data.astype('float32')[:, [0, 2]]
        #     data /= 15  # stdev plus a little ##Changed This
        #     self.range = 2
        #     return torch.FloatTensor(data).to(device) * self.scale

# if __name__ == '__main__':
#     np.set_printoptions(precision=2, suppress=True)
#     dtest = ToyDataset(distr="25Gaussians", scale=1.0, ratio=0.9)
#     # dtest.centeroids = dtest.centeroids[dtest.centeroids[:,1].argsort()]
#     print("dtest.centeroids.shape\n", dtest.centeroids.shape)
#     print("dtest.centeroids\n", dtest.centeroids)
#     samples = dtest.next_batch(100)
#     samples = samples[samples[:, 1].argsort()]
#     print("next_batch\n", samples)
