import typing as T
from array import array

import torch
import numpy as np

from tb_gen_data.config import OUTPUT_FOLDER
from tb_gen_data.gen_load_weights_base import GenLoadWeightsBase
from tb_gen_data.models.spike_deepclassifier import SpikeDeepClassifier
from tb_gen_data.utils import gen_mem_overwrite


class GenSpikeDeepClassifier(GenLoadWeightsBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        super().__init__(name, mem_weights_file="matlab_bar.txt", out_folder=out_folder)

        self.in_y = 20
        self.in_x = 48

        self.model = SpikeDeepClassifier()

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
        self.model.eval()
        self.output = self.model(self.input_)

    def _gen_mem_pre(self):
        """
        Get the memory before doing any computations.
        """
        spike_deeptector_params = self._get_model_params(self.model.spike_deeptector)
        bar_params = self._get_model_params(self.model.bar)

        mem_0, mem_1 = gen_mem_overwrite(
            self.input_ + self.model.spike_deeptector.outputs + self.model.bar.outputs
        )

        mem_0 = torch.zeros_like(mem_0)
        mem_1 = torch.zeros_like(mem_1)
        output = [torch.zeros_like(sample) for sample in self.output]

        tensors = (
            spike_deeptector_params + bar_params + self.input_ + [mem_0, mem_1] + output
        )
        flat_tensors = [torch.flatten(tensor) for tensor in tensors]

        self.mem_pre = [f"{x}\n" for tensor in flat_tensors for x in tensor.tolist()]

    def _gen_mem(self):
        spike_deeptector_params = self._get_model_params(self.model.spike_deeptector)
        bar_params = self._get_model_params(self.model.bar)

        mem_0, mem_1 = gen_mem_overwrite(
            self.input_ + self.model.spike_deeptector.outputs + self.model.bar.outputs
        )

        tensors = (
            spike_deeptector_params
            + bar_params
            + self.input_
            + [mem_0, mem_1]
            + self.output
        )
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
        print(f"int mem_1_len = {len(torch.flatten(mem_1))};")

    def _gen_mem_bin(self):
        FLOAT_SIZE_BYTES = 4
        BYTE_SIZE_BITS = 8
        BIT_ALIGNMENT = 512

        self.mem_bin = array("f")

        # this is notn very efficient, in production you probably would want
        # to get `self._mem_bin` first and then construct `self.mem` from it
        mem_list = [float(entry) for entry in self.mem]

        self.mem_bin.fromlist(mem_list)

        # memory needs alignment to BIT_ALIGNMENT bits
        mem_bin_size = len(self.mem_bin) * FLOAT_SIZE_BYTES * BYTE_SIZE_BITS
        pad_bits = (
            BIT_ALIGNMENT - mem_bin_size % BIT_ALIGNMENT
            if mem_bin_size % BIT_ALIGNMENT != 0
            else 0
        )

        if pad_bits % FLOAT_SIZE_BYTES * BYTE_SIZE_BITS != 0:
            print(
                f"Warning! Padding bits {pad_bits} can't be realized with 4 byte floats"
            )

        pad_floats = pad_bits // (FLOAT_SIZE_BYTES * BYTE_SIZE_BITS)

        self.mem_bin.fromlist([0.0 for _ in range(pad_floats)])


if __name__ == "__main__":
    torch.manual_seed(0)

    gen = GenSpikeDeepClassifier("spike_deepclassifier")
    gen.gen_input()
    gen._load_mem_file("matlab_spike_deeptector.txt", gen.model.spike_deeptector)
    gen._load_mem_file("matlab_bar.txt", gen.model.bar)
    gen.gen_output()
    gen.write_mem_pre()
    gen.write_mem()
    gen.write_mem_bin()
