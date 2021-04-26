from torch import nn

from config import IN_D, OUT_D, KERNEL_SIZE, STRIDE


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=IN_D, out_channels=OUT_D, kernel_size=KERNEL_SIZE, stride=STRIDE
        )

    def forward(self, x):
        return self.conv(x)
