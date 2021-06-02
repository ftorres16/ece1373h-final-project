import torch.nn as nn


class ConvBatchRelu(nn.Module):
    def __init__(self, in_d, out_d, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Conv2d(in_d, out_d, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_d, affine=False, momentum=None)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.y0 = self.conv(x)
        self.y1 = self.batch_norm(self.y0)
        self.y2 = self.relu(self.y1)

        return self.y2
