import typing as T

# import onnx
import torch
import torch.nn as nn

from config import OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase
from tb_gen_data.models.full_nn import MatlabCNN
from tb_gen_data.models.zero_mean import ZeroMean
from tb_gen_data.utils import gen_mem_overwrite


class GenMatlabCNN(GenBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        super().__init__(name, out_folder)

        self.model = MatlabCNN()

    def gen_input(
        self, batch_size: int = 1, in_d: int = 1, in_y: int = 20, in_x: int = 48
    ):
        self.input_ = torch.randn(batch_size, in_d, in_y, in_x)

    def gen_output(self):
        self.model.eval()
        self.output = self.model(self.input_)

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

    def _get_model_params(self) -> T.List[torch.tensor]:
        """
        Get all model params as a list.
        """
        params = []

        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, ZeroMean):
                params.append(layer.mean)
            elif isinstance(layer, nn.Conv2d):
                params.append(layer.weight)
                params.append(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                params.append(layer.running_mean)
                params.append(layer.running_var)
                params.append(
                    layer.weight
                    if layer.weight is not None
                    else torch.tensor([1.0] * layer.num_features)
                )
                params.append(
                    layer.bias
                    if layer.bias is not None
                    else torch.tensor([0.0] * layer.num_features)
                )

        return params

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

    gen = GenMatlabCNN("full_nn")
    gen.gen_input()
    gen.gen_output()
    gen.write_mem_pre()
    gen.write_mem()
    gen.write_params()
