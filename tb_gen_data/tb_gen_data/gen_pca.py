import torch

from tb_gen_data.config import IN_X, OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase
from tb_gen_data.models.pca import PCAModel


class GenPCA(GenBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        super().__init__(name, out_folder)

        self.model = PCAModel()

    def gen_input(
        self,
        in_x: int = IN_X,
        in_y: int = 48,
    ):
        self.input_ = torch.randn(in_x, in_y)

    def gen_output(self):
        self.output = self.model(self.input_)

    def _gen_mem(self):
        flat_tensors = [torch.flatten(self.input_), torch.flatten(self.output)]
        self.mem = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

    def _gen_params(self):
        self.params = {"in_rows": self.input_.shape[1]}


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenPCA("pca")

    gen.gen_input()
    gen.gen_output()
    gen.write_mem()
    gen.write_params()
