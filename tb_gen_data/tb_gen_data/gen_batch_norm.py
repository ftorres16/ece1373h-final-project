import torch
import torch.nn as nn

from tb_gen_data.config import BATCH_SIZE, IN_D, IN_X, IN_Y, OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase


class GenBatchNorm2D(GenBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER, in_d: int = IN_D):
        super().__init__(name, out_folder)

        self.batch_norm = nn.BatchNorm2d(in_d, affine=False, momentum=None)

    def gen_input(
        self,
        batch_size: int = BATCH_SIZE,
        in_d: int = IN_D,
        in_x: int = IN_X,
        in_y: int = IN_Y,
    ):
        self.input_ = torch.randn(batch_size, in_d, in_y, in_x)

    def gen_output(self):
        # run it at least once to initialize running var and mean
        _ = self.batch_norm(self.input_)

        self.batch_norm.eval()
        self.output = self.batch_norm(self.input_)

    def _gen_mem(self):
        weight = (
            self.batch_norm.weight
            if self.batch_norm.weight is not None
            else torch.tensor([1.0] * IN_D)
        )
        bias = (
            self.batch_norm.bias
            if self.batch_norm.bias is not None
            else torch.tensor([0.0] * IN_D)
        )

        flat_tensors = [
            self.batch_norm.running_mean,
            # store std dev instead of variance for hw efficiency
            torch.sqrt(self.batch_norm.running_var),
            weight,
            bias,
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
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenBatchNorm2D("batch_norm_2d")

    gen.gen_input()
    gen.gen_output()
    gen.write_mem()
    gen.write_params()
