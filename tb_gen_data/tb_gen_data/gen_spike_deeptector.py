import typing as T

import torch
from torch import nn

from config import OUTPUT_FOLDER
from tb_gen_data.gen_load_weights_base import GenLoadWeightsBase
from tb_gen_data.models.spike_deeptector import SpikeDeeptector
from tb_gen_data.models.zero_mean import ZeroMean
from tb_gen_data.utils import gen_mem_overwrite


class GenSpikeDeeptector(GenLoadWeightsBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        super().__init__(
            name, mem_weights_file="matlab_spike_deeptector.txt", out_folder=out_folder
        )

        self.in_y = 20
        self.in_x = 48

        self.model = SpikeDeeptector()

    def gen_input(
        self,
        batch_size: int = 1,
        in_d: int = 1,
        in_y: T.Optional[int] = None,
        in_x: T.Optional[int] = None,
    ):
        if in_y is None:
            in_y = self.in_y
        if in_x is None:
            in_x = self.in_x

        # self.input_ = torch.randn(batch_size, in_d, in_y, in_x)
        self.input_ = torch.zeros(batch_size, in_d, in_y, in_x)

    def gen_output(self):
        self.model.eval()
        self.output = self.model(self.input_)

        # # test exporting for matlab
        # aux = self.model.outputs[7]
        # aux = torch.squeeze(aux, 0)
        # aux = aux.permute(1, 2, 0)
        # aux = aux.numpy()
        # import scipy.io
        # scipy.io.savemat("test.mat", dict(aux_py=aux))

    def _get_outputs(self) -> T.List[torch.tensor]:
        """
        Get all model outputs as a list.
        """
        outputs = []

        for idx, layer in enumerate(self.model.layers):
            if (
                isinstance(layer, ZeroMean)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.MaxPool2d)
                or idx == len(self.model.layers) - 1
            ):
                # print(layer)
                outputs.append(self.model.outputs[idx])

        return outputs

    def _gen_mem_pre(self):
        """
        Get the memory before doing any computations.
        """
        params = self._get_model_params()
        outputs = self._get_outputs()

        mem_0, mem_1 = gen_mem_overwrite([self.input_] + outputs)

        mem_0 = torch.zeros_like(mem_0)
        mem_0[: len(torch.flatten(self.input_))] = torch.flatten(self.input_)

        mem_1 = torch.zeros_like(mem_1)

        tensors = params + [mem_0, mem_1]
        flat_tensors = [torch.flatten(tensor) for tensor in tensors]

        self.mem_pre = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

        # print(f"params len: {sum(len(torch.flatten(param)) for param in params)}")
        # print(f"mem_0 len: {len(torch.flatten(mem_0))}")
        # print(f"mem_1 len: {len(torch.flatten(mem_1))}")
        # print(f"pre mem len: {len(self.mem_pre)}")

    def _gen_mem(self):
        params = self._get_model_params()
        outputs = self._get_outputs()

        mem_0, mem_1 = gen_mem_overwrite([self.input_] + outputs)

        tensors = params + [mem_0, mem_1]

        flat_tensors = [torch.flatten(tensor) for tensor in tensors]

        self.mem = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

        print(f"int mem_len = {len(self.mem)};")
        print(f"int num_params = {sum(len(torch.flatten(param)) for param in params)};")
        print(f"int mem_0_len = {len(torch.flatten(mem_0))};")
        # print(f"mem_1 len: {len(torch.flatten(mem_1))}")
        # print(f"input len: {len(torch.flatten(self.input_))}")
        # print(f"last output len: {len(torch.flatten(outputs[-1]))}")
        # print(
        #     "all outputs len: ",
        #     sum(len(torch.flatten(mem)) for mem in [mem_0, mem_1]),
        # )

    def _gen_params(self):
        self.params = {}

        for idx, layer in enumerate(self.model.layers):
            # input_ = getattr(self.model, f"y_{idx - 1}") if idx > 0 else self.input_
            input_ = self.model.outputs[idx - 1] if idx > 0 else self.input_
            # output = getattr(self.model, f"y_{idx}")

            _params = {}

            if isinstance(layer, nn.Conv2d):
                _params = {
                    "b": input_.shape[0],
                    "id": input_.shape[1],
                    "ix": input_.shape[3],
                    "iy": input_.shape[2],
                    "od": layer.weight.shape[0],
                    "s": layer.stride[0],
                    "kx": layer.weight.shape[3],
                    "ky": layer.weight.shape[2],
                    "px": layer.padding[1],
                    "py": layer.padding[0],
                }
            elif isinstance(layer, nn.BatchNorm2d):
                _params = {
                    "b": input_.shape[0],
                    "id": input_.shape[1],
                    "ix": input_.shape[3],
                    "iy": input_.shape[2],
                }
            elif isinstance(layer, nn.ReLU):
                _params = {
                    "b": input_.shape[0],
                    "id": input_.shape[1],
                    "ix": input_.shape[3],
                    "iy": input_.shape[2],
                }

            elif isinstance(layer, nn.MaxPool2d):
                _params = {
                    "b": input_.shape[0],
                    "id": input_.shape[1],
                    "ix": input_.shape[3],
                    "iy": input_.shape[2],
                }

            _params = {f"{idx}_{key}": val for key, val in _params.items()}

            self.params.update(_params)


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenSpikeDeeptector("spike_deeptector")
    gen.gen_input()
    gen._load_mem_file()
    gen.gen_output()
    gen.write_mem_pre()
    gen.write_mem()
    gen.write_params()
