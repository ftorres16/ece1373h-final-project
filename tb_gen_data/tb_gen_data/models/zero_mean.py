import torch
import torch.nn as nn


class ZeroMean(nn.Module):
    def __init__(self, mean: torch.tensor = None):
        super(ZeroMean, self).__init__()

        self.mean = mean

    def forward(self, input):
        return input - self.mean
