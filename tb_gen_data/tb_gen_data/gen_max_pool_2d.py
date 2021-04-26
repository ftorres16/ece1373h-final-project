import torch
import torch.nn as nn

from tb_gen_data.config import BATCH_SIZE, KERNEL_SIZE, IN_D, IN_Y, IN_X, OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase


class GenMaxPool(GenBase):
    def __init__(
        self,
        out_file: str,
        kernel_size: int = KERNEL_SIZE,
    ):
        super().__init__(out_file)

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

    def write_output(self):
        with open(self.out_file, "w") as f:
            in_str = [f"{in_px}\n" for in_px in torch.flatten(self.input_).tolist()]
            f.writelines(in_str)

            out_str = [f"{out_px}\n" for out_px in torch.flatten(self.output).tolist()]
            f.writelines(out_str)


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenMaxPool(f"{OUTPUT_FOLDER}/max_pool_2d.txt")

    gen.gen_input()
    gen.gen_output()
    gen.write_output()
