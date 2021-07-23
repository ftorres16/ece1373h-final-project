import torch
import torch.nn as nn

from tb_gen_data.models.zero_mean import ZeroMean


class MatlabCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = ZeroMean(mean=torch.randn(1, 1, 20, 48))

        self.conv_1 = nn.Conv2d(
            1, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
        )
        self.batchnorm_1 = nn.BatchNorm2d(
            25,
            affine=False,
            momentum=None,
            # eps=10e-5,
            # momentum=0.1,
            # affine=True,
            # track_running_stats=True,
        )
        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Conv2d(25, 25, kernel_size=(20, 3), stride=(1, 1))
        self.batchnorm_2 = nn.BatchNorm2d(
            25,
            affine=False,
            momentum=None,
            # eps=10e-5,
            # momentum=0.1,
            # affine=True,
            # track_running_stats=True,
        )
        self.relu_2 = nn.ReLU()
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=(1, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            ceil_mode=False,
        )

        self.conv_3 = nn.Conv2d(25, 50, kernel_size=(1, 3), stride=(1, 1))
        self.batchnorm_3 = nn.BatchNorm2d(
            50,
            affine=False,
            momentum=None,
            # eps=10e-5,
            # momentum=0.1,
            # affine=True,
            # track_running_stats=True
        )
        self.relu_3 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(
            kernel_size=(1, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            ceil_mode=False,
        )

        self.conv_4 = nn.Conv2d(50, 100, kernel_size=(1, 3), stride=(1, 1))
        self.batchnorm_4 = nn.BatchNorm2d(
            100,
            affine=False,
            momentum=None,
            # eps=10e-5,
            # momentum=0.1,
            # affine=True,
            # track_running_stats=True,
        )
        self.relu_4 = nn.ReLU()
        self.max_pool_3 = nn.MaxPool2d(
            kernel_size=(1, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            ceil_mode=False,
        )

        self.conv_5 = nn.Conv2d(100, 100, kernel_size=(1, 5), stride=(1, 1))
        self.batchnorm_5 = nn.BatchNorm2d(
            100,
            affine=False,
            momentum=None,
            # eps=10e-5,
            # momentum=0.1,
            # affine=True,
            # track_running_stats=True,
        )
        self.relu_5 = nn.ReLU()
        self.max_pool_4 = nn.MaxPool2d(
            kernel_size=(1, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            ceil_mode=False,
        )

        self.conv_6 = nn.Conv2d(100, 100, kernel_size=(1, 5), stride=(1, 1))
        self.batchnorm_6 = nn.BatchNorm2d(
            100,
            affine=False,
            momentum=None,
            # eps=10e-5,
            # momentum=0.1,
            # affine=True,
            # track_running_stats=True,
        )
        self.relu_6 = nn.ReLU()
        self.max_pool_5 = nn.MaxPool2d(
            kernel_size=(1, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            ceil_mode=False,
        )

        self.conv_fc_1 = nn.Conv2d(100, 100, kernel_size=(1, 29), stride=(1, 1))
        self.relu_7 = nn.ReLU()
        self.conv_fc_2 = nn.Conv2d(100, 2, kernel_size=(1, 1), stride=(1, 1))
        # self.flatten = nn.Flatten()
        # self.softmax = nn.Softmax(dim=1)

        self.layers = [
            self.input_layer,
            self.conv_1,
            self.batchnorm_1,
            self.relu_1,
            self.conv_2,
            self.batchnorm_2,
            self.relu_2,
            self.max_pool_1,
            self.conv_3,
            self.batchnorm_3,
            self.relu_3,
            self.max_pool_2,
            self.conv_4,
            self.batchnorm_4,
            self.relu_4,
            self.max_pool_3,
            self.conv_5,
            self.batchnorm_5,
            self.relu_5,
            self.max_pool_4,
            self.conv_6,
            self.batchnorm_6,
            self.relu_6,
            self.max_pool_5,
            self.conv_fc_1,
            self.relu_7,
            self.conv_fc_2,
            # self.flatten,
            # self.softmax,
        ]

        self.outputs = []

    def forward(self, x):
        self.outputs = []

        for idx, func in enumerate(self.layers):
            in_ = self.outputs[-1] if idx > 0 else x

            self.outputs.append(func(in_))

        return self.outputs[-1]
