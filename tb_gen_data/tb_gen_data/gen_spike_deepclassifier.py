import typing as T
from array import array

import torch
import numpy as np

from tb_gen_data.config import OUTPUT_FOLDER
from tb_gen_data.gen_load_weights_base import GenLoadWeightsBase
from tb_gen_data.models.spike_deepclassifier import SpikeDeepClassifier
from tb_gen_data.utils import gen_mem_overwrite
from tb_gen_data.mem.mem import Mem


class GenSpikeDeepClassifier(GenLoadWeightsBase):
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        super().__init__(name, mem_weights_file="matlab_bar.txt", out_folder=out_folder)

        self.in_y = 20
        self.in_x = 48

        self.model = SpikeDeepClassifier()
        self.mem_obj = None

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
        self._gen_mem()
        self.mem_pre = self.mem_obj.to_text(pre=True)

    def _gen_mem(self):
        if getattr(self, "mem_obj") is not None:
            print(
                "Warning: Memory has already been generated, you should probably be OK."
                " Skipping this call."
            )
            return

        spike_deeptector_params = self._get_model_params(self.model.spike_deeptector)
        bar_params = self._get_model_params(self.model.bar)

        mem_0, mem_1 = gen_mem_overwrite(
            self.input_ + self.model.spike_deeptector.outputs + self.model.bar.outputs
        )

        input_len = sum(torch.numel(t) for t in self.input_)
        output_len = sum(torch.numel(t) for t in self.output)
        output_pad_zeros = input_len - output_len

        mem = Mem()
        _ = mem.add_tensor_list_chunk(
            spike_deeptector_params, "spike_deeptector_params"
        )
        _ = mem.add_tensor_list_chunk(bar_params, "bar_params")
        _ = mem.add_tensor_list_chunk(self.input_, "input")
        _ = mem.add_tensor_list_chunk(mem_0, "mem_0", masked=True)
        _ = mem.add_tensor_list_chunk(mem_1, "mem_1", masked=True)
        _ = mem.add_tensor_list_chunk(
            self.output, "output", pad_zeros=output_pad_zeros, masked=False
        )

        self.mem_obj = mem
        self.mem = mem.to_text()

        print(f"int mem_len = {mem.len};")
        print("int num_spike_deeptector_params" f" = {mem.chunks[0].len};")
        print("int num_bar_params" f" = {mem.chunks[1].len};")
        print(f"int input_len = {mem.chunks[2].len};")
        print(f"int mem_0_len = {mem.chunks[3].len};")
        print(f"int mem_1_len = {mem.chunks[4].len};")

    def _gen_mem_bin(self):
        self._gen_mem()
        self.mem_bin = self.mem_obj.to_bin()


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
