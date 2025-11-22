import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_full_grad_list(model, trainloader, optimizer, batch_size, device):
    parameters = [p for p in model.parameters()]
    full_grad_list = []
    init = True

    for xx, yy in trainloader:
        xx, yy = xx.to(device), yy.to(device)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss(reduction='mean')(model(xx), yy)
        loss.backward()

        if init:
            for params in parameters:
                full_grad = torch.zeros_like(params.grad.detach().data)
                full_grad_list.append(full_grad)
            init = False

        for i, params in enumerate(parameters):
            g = params.grad.detach().data
            full_grad_list[i] += (batch_size / len(trainloader)) * g

    total_norm = sum(grad.norm().item() ** 2 for grad in full_grad_list) ** 0.5
    return total_norm

def sampling_replacement(batch_size, roll, q):
    samples = np.random.choice(batch_size, size=roll, replace=True, p=q)
    S = np.bincount(samples, minlength=batch_size)
    v = S / (roll * q)
    return v

def sampling_independent(batch_size, p):
    S = np.random.binomial(1, p)
    v = np.zeros(batch_size)
    for i in range(batch_size):
        if S[i] == 1:
            v[i] = 1 / p[i]
    return v

def sampling_tau_nice(batch_size, tau):
    p = np.array([tau / batch_size] * batch_size)
    return sampling_independent(batch_size, p)