import torch.nn as nn

from config import IN_Y, OUT_Y


class FC(nn.Module):
    def __init__(self, in_features: int = IN_Y, out_features: int = OUT_Y):
        super().__init__()
        self.linear = nn.Linear(IN_Y, OUT_Y)

    def forward(self, x):
        return self.linear(x)
