import onnx
import torch
from onnx2pytorch import ConvertModel

from config import OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase


class GenMatlabCNN(GenBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        super().__init__(name, out_folder)

        self.onnx_model = onnx.load("inputs/trainedModelDeepNetCNN.onnx")
        self.model = ConvertModel(self.onnx_model)


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenMatlabCNN("matlab_cnn")
