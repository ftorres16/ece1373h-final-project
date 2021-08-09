import torch
from torch import nn

from tb_gen_data.models.zero_mean import ZeroMean


class BAR(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = ZeroMean(mean=torch.randn(1, 1, 1, 48))

        self.conv_1 = nn.Conv2d(
            1, 50, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
        )
        self.batchnorm_1 = nn.BatchNorm2d(50, affine=False, momentum=None)
        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Conv2d(50, 50, kernel_size=(1, 5), stride=(1, 1))
        self.batchnorm_2 = nn.BatchNorm2d(50, affine=False, momentum=None)
        self.relu_2 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(
            kernel_size=(1, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            ceil_mode=False,
        )

        self.conv_3 = nn.Conv2d(50, 75, kernel_size=(1, 5), stride=(1, 1))
        self.batchnorm_3 = nn.BatchNorm2d(75, affine=False, momentum=None)
        self.relu_3 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(
            kernel_size=(1, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            ceil_mode=False,
        )

        self.fc_1 = nn.Conv2d(75, 600, kernel_size=(1, 38), stride=(1, 1))
        self.relu_4 = nn.ReLU()

        self.fc_2 = nn.Conv2d(600, 300, kernel_size=(1, 1), stride=(1, 1))
        self.relu_5 = nn.ReLU()

        self.fc_3 = nn.Conv2d(300, 2, kernel_size=(1, 1), stride=(1, 1))

        self.layers = [
            self.input_layer,
            self.conv_1,
            self.batchnorm_1,
            self.relu_1,
            self.conv_2,
            self.batchnorm_2,
            self.relu_2,
            self.maxpool_1,
            self.conv_3,
            self.batchnorm_3,
            self.relu_3,
            self.maxpool_2,
            self.fc_1,
            self.relu_4,
            self.fc_2,
            self.relu_5,
            self.fc_3,
        ]
        self.outputs = []

    def forward(self, x):
        self.outputs = []

        for idx, func in enumerate(self.layers):
            in_ = self.outputs[-1] if idx > 0 else x

            self.outputs.append(func(in_))

        return self.outputs[-1]
