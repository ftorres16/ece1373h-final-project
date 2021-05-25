import torch.nn as nn


class ConvRelu(nn.Module):
    def __init__(self, in_d, out_d, kernel_size, stride):
        super().__init__()

        self.conv = nn.Conv2d(in_d, out_d, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.y0 = self.conv(x)
        self.y1 = self.relu(self.y0)

        return self.y1
