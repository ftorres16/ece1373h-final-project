import torch
import torch.nn as nn

from config import BATCH_SIZE, IN_X, IN_Y, OUT_X, OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase


class GenFC(GenBase):
    def __init__(self, out_file: str):
        super().__init__(out_file)

        self.fc = nn.Linear(IN_X, OUT_X)

    def gen_input(self):
        self.input_ = torch.randn(BATCH_SIZE, IN_Y, IN_X)

    def gen_output(self):
        self.output = self.fc(self.input_)

    def write_output(self):
        with open(self.out_file, "w") as f:
            weights = [
                f"{weight}\n" for weight in torch.flatten(self.fc.weight).tolist()
            ]
            f.writelines(weights)

            biases = [f"{bias}\n" for bias in self.fc.bias.tolist()]
            f.writelines(biases)

            in_str = [f"{in_px}\n" for in_px in torch.flatten(self.input_).tolist()]
            f.writelines(in_str)

            out_str = [f"{out_px}\n" for out_px in torch.flatten(self.output).tolist()]
            f.writelines(out_str)


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenFC(f"{OUTPUT_FOLDER}/fc.txt")

    gen.gen_input()
    gen.gen_output()
    gen.write_output()
