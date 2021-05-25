import torch

from config import KERNEL_SIZE, IN_D, IN_Y, IN_X, OUT_D, OUTPUT_FOLDER, STRIDE
from tb_gen_data.gen_base import GenBase
from tb_gen_data.models.conv_relu import ConvRelu


class GenConvRelu(GenBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        super().__init__(name, out_folder)

        self.model = ConvRelu(
            in_d=IN_D, out_d=OUT_D, kernel_size=KERNEL_SIZE, stride=STRIDE
        )

    def gen_input(
        self, batch_size: int = 1, in_d: int = IN_D, in_y: int = IN_Y, in_x: int = IN_X
    ):
        self.input_ = torch.randn(batch_size, in_d, in_y, in_x)

    def gen_output(self):
        self.output = self.model(self.input_)

    def _gen_mem(self):
        tensors = []

        tensors.append(self.model.conv.weight)
        tensors.append(self.model.conv.bias)

        # input
        tensors.append(self.input_)

        # output
        tensors.append(self.model.y0)
        tensors.append(self.model.y1)

        flat_tensors = [torch.flatten(tensor) for tensor in tensors]

        self.mem = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

    def _gen_params(self):
        self.params = {
            "b": self.input_.shape[0],
            "id": self.input_.shape[1],
            "ix": self.input_.shape[3],
            "iy": self.input_.shape[2],
            "od": self.model.conv.weight.shape[0],
            "s": self.model.conv.stride[0],
            "k": self.model.conv.weight.shape[2],
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenConvRelu("conv_relu")
    gen.gen_input()
    gen.gen_output()
    gen.write_mem()
    gen.write_params()
