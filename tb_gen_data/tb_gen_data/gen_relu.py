import torch
import torch.nn as nn

from config import BATCH_SIZE, IN_X, IN_Y, OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase


class GenReLU(GenBase):
    def __init__(self, out_file: str):
        super().__init__(out_file)

        self.relu = nn.ReLU()

    def gen_input(self):
        self.input_ = torch.randn(BATCH_SIZE, IN_X, IN_Y)

    def gen_output(self):
        self.output = self.relu(self.input_)

    def write_output(self):
        with open(self.out_file, "w") as f:
            in_str = [f"{in_px}\n" for in_px in torch.flatten(self.input_).tolist()]
            f.writelines(in_str)

            out_str = [f"{out_px}\n" for out_px in torch.flatten(self.output).tolist()]
            f.writelines(out_str)


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenReLU(f"{OUTPUT_FOLDER}/relu.txt")

    gen.gen_input()
    gen.gen_output()
    gen.write_output()
