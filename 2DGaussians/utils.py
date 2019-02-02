import numpy as np
import torch
import torch.nn as nn
import torch.autograd as ag
from torch.autograd import Variable
import matplotlib.pyplot as plt

def draw_grad(in_batch, D, criterion, fig, ax, device):
    ones = torch.ones(in_batch.shape[0], 1, device=device)

    in_batch.requires_grad_()
    with torch.enable_grad():
        out_batch = D(in_batch)
        loss = criterion.forward(out_batch, ones)
        loss.backward()

    data = in_batch.data.cpu().numpy()
    grad = -in_batch.grad.cpu().numpy()

    ax.quiver(data[:, 0], data[:, 1], grad[:, 0], grad[:, 1], color='g')

