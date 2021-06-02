# import onnx
import torch
import torch.nn as nn

# from onnx2pytorch import ConvertModel

from config import OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase
from tb_gen_data.models.matlab_cnn import MatlabCNN


class GenMatlabCNN(GenBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        super().__init__(name, out_folder)

        # self.onnx_model = onnx.load("inputs/trainedModelDeepNetCNN.onnx")
        # self.model = ConvertModel(self.onnx_model)

        self.model = MatlabCNN()

    def gen_input(
        self, batch_size: int = 1, in_d: int = 1, in_y: int = 68, in_x: int = 50
    ):
        self.input_ = torch.randn(batch_size, in_d, in_y, in_x)

    def gen_output(self):
        # self.model.eval()
        self.output = self.model(self.input_)

    def _gen_mem(self):
        params = []
        outputs = []

        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, nn.Conv2d):
                params.append(layer.weight)
                params.append(layer.bias)

            if (
                isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.MaxPool2d)
                or idx == len(self.model.layers) - 1
            ):
                print(layer)
                outputs.append(self.model.outputs[idx])

        tensors = [self.input_] + params + outputs

        flat_tensors = [torch.flatten(tensor) for tensor in tensors]

        self.mem = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

        print(f"total len: {len(self.mem)}")
        print(f"input len: {len(torch.flatten(self.input_))}")
        print(f"params len: {sum(len(torch.flatten(param)) for param in params)}")
        print(
            "intermediate outputs len: ",
            sum(len(torch.flatten(output)) for output in outputs[:-1]),
        )
        print(f"last output len: {len(torch.flatten(outputs[-1]))}")
        print(
            "all outputs len: ",
            sum(len(torch.flatten(output)) for output in outputs),
        )

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
    gen.write_mem()
    gen.write_params()
