import torch

from tb_gen_data.config import BATCH_SIZE, IN_D, IN_X, IN_Y, OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase
from tb_gen_data.models.zero_mean import ZeroMean


class GenZeroMean(GenBase):
    def __init__(
        self,
        name: str,
        out_folder: str = OUTPUT_FOLDER,
        in_y: int = IN_Y,
        in_x: int = IN_X,
    ):
        super().__init__(name, out_folder)

        self.model = ZeroMean(mean=torch.randn(1, 1, in_y, in_x))

    def gen_input(
        self,
        batch_size: int = BATCH_SIZE,
        in_d: int = IN_D,
        in_x: int = IN_X,
        in_y: int = IN_Y,
    ):
        self.input_ = torch.randn(batch_size, in_d, in_y, in_x)

    def gen_output(self):
        self.output = self.model(self.input_)

    def _gen_mem(self):
        flat_tensors = [
            torch.flatten(self.model.mean),
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

    gen = GenZeroMean("zero_mean")

    gen.gen_input()
    gen.gen_output()
    gen.write_mem()
    gen.write_params()
