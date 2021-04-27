import torch
import torch.nn as nn

from config import BATCH_SIZE, IN_X, IN_Y, OUT_X, OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase


class GenFC(GenBase):
    def __init__(
        self,
        name: str,
        out_folder: str = OUTPUT_FOLDER,
        in_x: int = IN_X,
        out_x: int = OUT_X,
    ):
        super().__init__(name, out_folder)

        self.fc = nn.Linear(in_x, out_x)

    def gen_input(
        self, batch_size: int = BATCH_SIZE, in_y: int = IN_Y, in_x: int = IN_X
    ):
        self.input_ = torch.randn(batch_size, in_y, in_x)

    def gen_output(self):
        self.output = self.fc(self.input_)

    def _gen_mem(self):
        flat_tensors = [
            torch.flatten(self.fc.weight),
            self.fc.bias,
            torch.flatten(self.input_),
            torch.flatten(self.output),
        ]

        self.mem = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

    def _gen_params(self):
        self.params = {
            "b": self.input_.shape[0],
            "ix": self.input_.shape[2],
            "iy": self.input_.shape[1],
            "ox": self.output.shape[2],
            "oy": self.output.shape[1],
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenFC("fc")

    gen.gen_input()
    gen.gen_output()
    gen.write_mem()
    gen.write_params()
