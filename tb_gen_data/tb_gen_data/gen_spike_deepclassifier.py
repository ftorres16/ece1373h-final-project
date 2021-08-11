import typing as T

import torch
import numpy as np

from tb_gen_data.config import OUTPUT_FOLDER
from tb_gen_data.gen_load_weights_base import GenLoadWeightsBase
from tb_gen_data.models.bar import BAR
from tb_gen_data.models.spike_deeptector import SpikeDeeptector
from tb_gen_data.utils import gen_mem_overwrite


class GenSpikeDeepClassifier(GenLoadWeightsBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        super().__init__(name, mem_weights_file="matlab_bar.txt", out_folder=out_folder)

        self.in_y = 20
        self.in_x = 48

        self.spike_deeptector = SpikeDeeptector()
        self.bar = BAR()

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

        self.input_ = [
            torch.randn(2, in_d, in_y, in_x),
            # torch.randn(1, in_d, in_y, in_x),
        ]

        # add one sample from the real thing
        real_data = np.genfromtxt("inputs/waveform_0.txt", delimiter=",")
        real_data = real_data.T
        real_data = torch.tensor(real_data, dtype=torch.float32)
        real_data = real_data.reshape((1, 1, 20, 48))

        self.input_.append(real_data)

    def gen_output(self):
        self.spike_deeptector.eval()
        self.bar.eval()

        self.spike_deeptector_outputs = [
            self.spike_deeptector(input_) for input_ in self.input_
        ]
        self.bar_outputs = [
            self.bar(sample)
            for electrode in self.input_
            for chunk in torch.split(electrode, 1)
            for sample in torch.split(chunk, 1, dim=2)
        ]

    def _gen_mem(self):
        spike_deeptector_params = self._get_model_params(self.spike_deeptector)
        bar_params = self._get_model_params(self.bar)

        mem_0, mem_1 = gen_mem_overwrite(
            self.input_ + self.spike_deeptector.outputs + self.bar.outputs
        )

        tensors = spike_deeptector_params + bar_params + self.input_ + [mem_0, mem_1]
        flat_tensors = [torch.flatten(tensor) for tensor in tensors]

        self.mem = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

        print(f"int mem_len = {len(self.mem)};")
        print(
            "int num_spike_deeptector_params"
            f" = {sum(len(torch.flatten(param)) for param in spike_deeptector_params)};"
        )
        print(
            "int num_bar_params"
            f" = {sum(len(torch.flatten(param)) for param in bar_params)};"
        )
        print(
            f"int input_len = {sum(len(torch.flatten(input_)) for input_ in self.input_)};"
        )
        print(f"int mem_0_len = {len(torch.flatten(mem_0))};")


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenSpikeDeepClassifier("spike_deepclassifier")
    gen.gen_input()
    gen._load_mem_file("matlab_spike_deeptector.txt", gen.spike_deeptector)
    gen._load_mem_file("matlab_bar.txt", gen.bar)
    gen.gen_output()
    gen.write_mem()
