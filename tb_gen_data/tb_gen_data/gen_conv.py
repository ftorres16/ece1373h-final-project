import typing as T

import torch
import torch.nn as nn

from config import (
    BATCH_SIZE,
    KERNEL_SIZE,
    IN_D,
    IN_Y,
    IN_X,
    OUT_D,
    OUTPUT_FOLDER,
    PAD_X,
    PAD_Y,
    STRIDE,
)
from tb_gen_data.gen_base import GenBase


class GenConv(GenBase):
    def __init__(
        self,
        name: str,
        out_folder: str = OUTPUT_FOLDER,
        in_d: int = IN_D,
        out_d: int = OUT_D,
        kernel_size: int = KERNEL_SIZE,
        stride: int = STRIDE,
        padding: T.Tuple[int] = (PAD_X, PAD_Y),
    ):
        super().__init__(name, out_folder)

        self.conv = nn.Conv2d(
            in_channels=in_d,
            out_channels=out_d,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def gen_input(
        self,
        batch_size: int = BATCH_SIZE,
        in_d: int = IN_D,
        in_y: int = IN_Y,
        in_x: int = IN_X,
    ):
        self.input_ = torch.randn(batch_size, in_d, in_y, in_x)

    def gen_output(self):
        self.output = self.conv(self.input_)

    def _gen_mem(self):
        flat_tensors = [
            torch.flatten(self.conv.weight),
            self.conv.bias,
            torch.flatten(self.input_),
            torch.flatten(self.output),
        ]

        self.mem = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

    def _gen_params(self):
        self.params = {
            "b": self.input_.shape[0],
            "id": self.input_.shape[1],
            "ix": self.input_.shape[3],
            "iy": self.input_.shape[2],
            "od": self.conv.weight.shape[0],
            "s": self.conv.stride[0],
            "k": self.conv.weight.shape[2],
            "px": self.conv.padding[1],
            "py": self.conv.padding[0],
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenConv("conv")

    gen.gen_input()
    gen.gen_output()
    gen.write_mem()
    gen.write_params()
