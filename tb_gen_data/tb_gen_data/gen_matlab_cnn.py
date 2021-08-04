import typing as T

import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper

from config import OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase


class GenMatlabCNN(GenBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        super().__init__(name, out_folder)

        onnx_path = "inputs/trainedModelDeepNetCNN.onnx"

        self.onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(self.onnx_model)

        self.ort_session = onnxruntime.InferenceSession(onnx_path)

    def gen_input(
        self, batch_size: int = 1, in_d: int = 1, in_y: int = 20, in_x: int = 48
    ):
        # np_input = np.random.randn(batch_size, in_d, in_y, in_x).astype("f")
        np_input = np.zeros((batch_size, in_d, in_y, in_x)).astype("f")
        self.input_ = {self.ort_session.get_inputs()[0].name: np_input}

    def gen_output(self):
        [self.output] = self.ort_session.run(None, self.input_)
        self.output = np.ndarray.flatten(self.output)

    def _get_model_params(self) -> T.List[np.ndarray]:
        """
        Get all model params as a list.
        """
        layers = {
            layer.name: numpy_helper.to_array(layer)
            for layer in self.onnx_model.graph.initializer
        }

        param_names = [
            "imageinput_Mean",
            "conv_1_W",
            "conv_1_B",
            "batchnorm_1_mean",
            "batchnorm_1_var",
            "batchnorm_1_scale",
            "batchnorm_1_B",
            "conv_2_W",
            "conv_2_B",
            "batchnorm_2_mean",
            "batchnorm_2_var",
            "batchnorm_2_scale",
            "batchnorm_2_B",
            "conv_3_W",
            "conv_3_B",
            "batchnorm_3_mean",
            "batchnorm_3_var",
            "batchnorm_3_scale",
            "batchnorm_3_B",
            "conv_4_W",
            "conv_4_B",
            "batchnorm_4_mean",
            "batchnorm_4_var",
            "batchnorm_4_scale",
            "batchnorm_4_B",
            "conv_5_W",
            "conv_5_B",
            "batchnorm_5_mean",
            "batchnorm_5_var",
            "batchnorm_5_scale",
            "batchnorm_5_B",
            "conv_6_W",
            "conv_6_B",
            "batchnorm_6_mean",
            "batchnorm_6_var",
            "batchnorm_6_scale",
            "batchnorm_6_B",
            "fc_1_W",
            "fc_1_B",
            "fc_2_W",
            "fc_2_B",
        ]

        params = [layers[param] for param in param_names]
        return params

    def _gen_mem_pre(self):
        """
        Get the memory before doing any computations.
        """
        params = self._get_model_params()

        # because we cant easily get intermediate values from ONNX runtime
        # use the pytorch value
        mem_0 = np.zeros(24_000)
        mem_1 = np.zeros(4_000)

        flat_input = np.ndarray.flatten(list(self.input_.values())[0])

        mem_0[: len(flat_input)] = flat_input

        tensors = params + [mem_0, mem_1]
        flat_tensors = [np.ndarray.flatten(tensor) for tensor in tensors]

        self.mem_pre = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

    def _gen_mem(self):
        params = self._get_model_params()
        self.gen_output()

        mem_0 = np.zeros(24_000)
        mem_1 = np.zeros(4_000)

        mem_0[: len(self.output)] = self.output

        tensors = params + [mem_0, mem_1]

        flat_tensors = [np.ndarray.flatten(tensor) for tensor in tensors]

        self.mem = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

        print(f"int mem_len = {len(self.mem)};")
        print(
            f"int num_params = {sum(len(np.ndarray.flatten(param)) for param in params)};"
        )
        print(f"int mem_0_len = {len(np.ndarray.flatten(mem_0))};")
        # print(f"mem_1 len: {len(np.ndarray.flatten(mem_1))}")
        # print(f"input len: {len(np.ndarray.flatten(self.input_))}")
        # print(f"last output len: {len(np.ndarray.flatten(outputs[-1]))}")
        # print(
        #     "all outputs len: ",
        #     sum(len(np.ndarray.flatten(mem)) for mem in [mem_0, mem_1]),
        # )


if __name__ == "__main__":
    np.random.seed(0)

    gen = GenMatlabCNN("matlab_nn")
    gen.gen_input()
    gen.gen_output()
    gen.write_mem_pre()
    gen.write_mem()
    # gen.write_params()
