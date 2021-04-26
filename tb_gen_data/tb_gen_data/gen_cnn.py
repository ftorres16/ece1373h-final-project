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
    STRIDE,
)
from tb_gen_data.gen_base import GenBase


class GenCNN(GenBase):
    def __init__(self, out_file: str):
        super().__init__(out_file)

        self.cnn = nn.Conv2d(
            in_channels=IN_D, out_channels=OUT_D, kernel_size=KERNEL_SIZE, stride=STRIDE
        )

    def gen_input(self):
        self.input_ = torch.randn(BATCH_SIZE, IN_D, IN_Y, IN_X)

    def gen_output(self):
        self.output = self.cnn(self.input_)

    def write_output(self):
        with open(self.out_file, "w") as f:
            weights = [
                f"{weight}\n" for weight in torch.flatten(self.cnn.weight).tolist()
            ]
            f.writelines(weights)

            biases = [f"{bias}\n" for bias in self.cnn.bias.tolist()]
            f.writelines(biases)

            in_str = [f"{in_px}\n" for in_px in torch.flatten(self.input_).tolist()]
            f.writelines(in_str)

            out_str = [f"{out_px}\n" for out_px in torch.flatten(self.output).tolist()]
            f.writelines(out_str)


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenCNN(f"{OUTPUT_FOLDER}/cnn.txt")

    gen.gen_input()
    gen.gen_output()
    gen.write_output()
