import typing as T

import torch

from config import (
    BATCH_SIZE,
    IN_D,
    IN_Y,
    IN_X,
    KERNEL_SIZE,
    PAD_X,
    PAD_Y,
    OUT_D,
    OUTPUT_FOLDER,
    STRIDE,
)
from tb_gen_data.gen_base import GenBase
from tb_gen_data.models.conv_batch_relu import ConvBatchRelu


class GenConvBatchRelu(GenBase):
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

        self.model = ConvBatchRelu(
            in_d=in_d,
            out_d=out_d,
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
        self.model.eval()
        self.output = self.model(self.input_)

    def _gen_mem(self):
        bn_weight = (
            self.model.batch_norm.weight
            if self.model.batch_norm.weight is not None
            else torch.tensor([1.0] * OUT_D)
        )
        bn_bias = (
            self.model.batch_norm.bias
            if self.model.batch_norm.bias is not None
            else torch.tensor([0.0] * OUT_D)
        )

        flat_tensors = [
            torch.flatten(self.model.conv.weight),
            self.model.conv.bias,
            self.model.batch_norm.running_mean,
            self.model.batch_norm.running_var,
            bn_weight,
            bn_bias,
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
            "od": self.model.conv.weight.shape[0],
            "s": self.model.conv.stride[0],
            "kx": self.model.conv.weight.shape[3],
            "ky": self.model.conv.weight.shape[2],
            "px": self.model.conv.padding[1],
            "py": self.model.conv.padding[0],
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenConvBatchRelu("conv_batch_relu")

    gen.gen_input()
    gen.gen_output()
    gen.write_mem()
    gen.write_params()
