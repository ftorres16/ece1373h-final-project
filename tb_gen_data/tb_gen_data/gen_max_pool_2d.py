import torch
import torch.nn as nn

from tb_gen_data.config import BATCH_SIZE, KERNEL_SIZE, IN_D, IN_Y, IN_X, OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase


class GenMaxPool2D(GenBase):
    def __init__(
        self,
        name: str,
        out_folder: str = OUTPUT_FOLDER,
        kernel_size: int = KERNEL_SIZE,
    ):
        super().__init__(name, out_folder)

        self.max_pool = nn.MaxPool2d(kernel_size)

    def gen_input(
        self,
        batch_size: int = BATCH_SIZE,
        in_d: int = IN_D,
        in_x: int = IN_X,
        in_y: int = IN_Y,
    ):
        self.input_ = torch.randn(batch_size, in_d, in_y, in_x)

    def gen_output(self):
        self.output = self.max_pool(self.input_)

    def _gen_mem(self):
        flat_tensors = [
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
            "k": self.max_pool.kernel_size,
            "s": self.max_pool.stride,
            "od": self.output.shape[1],
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenMaxPool2D("max_pool_2d")

    gen.gen_input()
    gen.gen_output()
    gen.write_mem()
    gen.write_params()
