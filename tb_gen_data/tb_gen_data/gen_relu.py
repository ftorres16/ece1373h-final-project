import torch
import torch.nn as nn

from config import BATCH_SIZE, IN_X, IN_Y, OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase


class GenReLU(GenBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        super().__init__(name, out_folder)

        self.relu = nn.ReLU()

    def gen_input(
        self, batch_size: int = BATCH_SIZE, in_x: int = IN_X, in_y: int = IN_Y
    ):
        self.input_ = torch.randn(batch_size, in_x, in_y)

    def gen_output(self):
        self.output = self.relu(self.input_)

    def _gen_mem(self):
        flat_tensors = [torch.flatten(self.input_), torch.flatten(self.output)]

        self.mem = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

    def _gen_params(self):
        self.params = {
            "b": self.input_.shape[0],
            "ix": self.input_.shape[1],
            "iy": self.input_.shape[2],
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenReLU("relu")

    gen.gen_input()
    gen.gen_output()
    gen.write_mem()
    gen.write_params()
