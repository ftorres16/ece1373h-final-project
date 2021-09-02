import torch
from torch import nn


class PCAModel(nn.Module):
    def __init__(self):
        super(PCAModel, self).__init__()

    def forward(self, input):
        demeaned_input = input - torch.mean(input, dim=0, keepdim=True)

        cov = torch.matmul(demeaned_input.T, demeaned_input)
        cov /= input.shape[0] - 1

        U, S, V = torch.svd(cov, some=False)

        return torch.matmul(input, V)
