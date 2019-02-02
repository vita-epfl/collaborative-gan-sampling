import numpy as np
import torch
import random
from torchvision import transforms, datasets
import sklearn.datasets
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
np.random.seed(1234)
torch.manual_seed(999)

class NoiseDataset:
    def __init__(self, distr='Gaussian', dim=2, mean=(0, 0), var=1):
        self.distr = distr
        self.dim = dim
        # self.mean = torch.FloatTensor(mean)
        self.var = var

    def next_batch(self, batch_size=64, device=None):
        if self.distr == 'Gaussian':
            return torch.randn(batch_size, self.dim, device=device)  # + self.mean.expand(batch_size, self.dim).to(device)
        else:
            return 'Not supported distribution'


class ToyDataset:
    def __init__(self, distr='8Gaussians', dim=2, scale=2, ratio=0.5):
        self.distr = distr
        self.dim = dim
        self.scale = scale
        self.ratio = ratio

        self.dataset = []
        for i in range(100000 // 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    self.dataset.append(point)
        self.dataset = np.array(self.dataset, dtype='float32')
        np.random.shuffle(self.dataset)
        self.dataset /= 2.828  # stdev

        self.range = 1
        if self.distr == 'swissroll':
            self.range = 2

    def next_batch(self, batch_size=64, device=None, ratio=None):
        if self.distr == '8Gaussians':
            centers = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1. / np.sqrt(2), 1. / np.sqrt(2)),
                (1. / np.sqrt(2), -1. / np.sqrt(2)),
                (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                (-1. / np.sqrt(2), -1. / np.sqrt(2))
            ]
            centers = [(self.scale * x, self.scale * y) for x, y in centers]
            dataset = []
            for i in range(batch_size):
                point = np.random.randn(2) * .05
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev

            return torch.FloatTensor(dataset).to(device)

        if self.distr == 'Imbal-8Gaussians':
            centers = [
                (1, 0),
                (1. / np.sqrt(2), 1. / np.sqrt(2)),
                (1. / np.sqrt(2), -1. / np.sqrt(2)),
                (-1, 0),
                (0, 1),
                (0, -1),
                (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                (-1. / np.sqrt(2), -1. / np.sqrt(2))
            ]
            centers = [(self.scale * x, self.scale * y) for x, y in centers]
            dataset = []
            for i in range(batch_size):
                point = np.random.randn(2) * .05
                if np.random.rand(1,1) < self.ratio:
                    center = random.choice(centers[:2])
                else:
                    center = random.choice(centers[2:])
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev

            return torch.FloatTensor(dataset).to(device)

        if self.distr == '25Gaussians':
            batch_idx = np.random.randint(100000 // batch_size)
            return torch.FloatTensor(self.dataset[batch_idx * batch_size:(batch_idx + 1) * batch_size]).to(device) * self.scale

        if self.distr == 'swissroll':
            data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=0.25)[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 15  # stdev plus a little ##Changed This
            self.range = 2
            return torch.FloatTensor(data).to(device) * self.scale


class MNISTDataset:
    def __init__(self, train=True):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/mnist', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=1, shuffle=True)

        self.data = []
        for i, (x, y) in enumerate(self.train_loader):
            self.data.append(x.view(1, -1))
        self.loc = 0
        self.n_samples = len(self.data)

    def next_batch(self, batch_size=32, device=None):
        if self.loc + batch_size > self.n_samples:
            random.shuffle(self.data)
            self.loc = 0

        batch = self.data[self.loc : self.loc + batch_size]
        self.loc += batch_size
        batch = torch.cat(batch, 0)
        return batch.to(device)


class Cifar10Dataset:
    def __init__(self):
        dataset = datasets.CIFAR10(root='./data/cifar10', download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(32),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        self.train_loader = torch.utils.data.DataLoader(dataset)
        self.data = []
        for i, (x, y) in enumerate(self.train_loader):
            self.data.append(x.view(1, -1))
        self.loc = 0
        self.n_samples = len(self.data)

    def next_batch(self, batch_size=32, device=None):
        if self.loc + batch_size > self.n_samples:
            random.shuffle(self.data)
            self.loc = 0

        batch = self.data[self.loc : self.loc + batch_size]
        self.loc += batch_size
        batch = torch.cat(batch, 0)
        return batch.to(device)


class SequentialMNISTDataset:
    def __init__(self, shuffle=True, repeat=2):
        self.data = []
        self.cur_idx = 0
        self.cur_label = 0
        self.repeat = repeat
        self.repeat_count = 0
        self.shuffle = shuffle

        for _ in range(10):
            self.data.append([])

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=1, shuffle=True)
        # self.test_loader = torch.utils.data.DataLoader(
        #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ])),
        #     batch_size=1, shuffle=True)

        self.__sequentialize()

    def __sequentialize(self):
        for i, (img, label) in enumerate(self.train_loader):
            self.data[label].append(img.view(1, -1))

    def __shuffle(self, label=-1):
        if label < 0:
            for d in self.data:
                random.shuffle(d)
        else:
            random.shuffle(self.data[label])

    def next_batch(self, batch_size, device=None):
        next_idx = self.cur_idx + batch_size
        if next_idx > len(self.data[self.cur_label]):
            if self.repeat_count < self.repeat:
                print('repeat %d/%d' % (self.repeat_count, self.repeat))
                self.repeat_count += 1
                self.cur_idx = 0
                if self.shuffle:
                    self.__shuffle(self.cur_label)
            else:
                self.repeat_count = 0
                print('switch from label %d to label %d' % (self.cur_label, self.cur_label + 1))
                if self.cur_label == 9:
                    if self.shuffle:
                        self.__shuffle()
                    self.cur_label = 0

                self.cur_label += 1
                self.cur_idx = 0

        next_idx = self.cur_idx + batch_size
        # ret = [d.unsqueeze(0) for d in self.data[self.cur_label][self.cur_idx:next_idx]]
        ret = self.data[self.cur_label][self.cur_idx:next_idx]
        ret = torch.cat(ret, 0)
        self.cur_idx = next_idx
        return ret.to(device)