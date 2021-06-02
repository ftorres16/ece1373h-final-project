import torch.nn as nn


class ConvBatchReluMax(nn.Module):
    def __init__(
        self,
        in_d,
        out_d,
        kernel_size,
        stride,
        padding,
        max_pool_kernel,
        max_pool_stride,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_d, out_d, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_d, affine=False, momentum=None)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(
            kernel_size=max_pool_kernel,
            stride=max_pool_stride,
            padding=(0, 0),
            dilation=1,
            ceil_mode=False,
        )

    def forward(self, x):
        self.y0 = self.conv(x)
        self.y1 = self.batch_norm(self.y0)
        self.y2 = self.relu(self.y1)
        self.y3 = self.max_pool(self.y2)

        return self.y3
