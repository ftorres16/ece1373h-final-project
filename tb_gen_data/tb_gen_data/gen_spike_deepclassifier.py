import typing as T

import torch
import numpy as np

from tb_gen_data.config import OUTPUT_FOLDER
from tb_gen_data.gen_load_weights_base import GenLoadWeightsBase
from tb_gen_data.models.spike_deepclassifier import SpikeDeepClassifier
from tb_gen_data.utils import gen_mem_overwrite
from tb_gen_data.mem.mem import Mem

NUM_ELECTRODES = 96
FLOAT_SIZE_BYTES = 4


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
        _ = self.model(self.input_)

        self.pca_outputs = self.model.pca_outputs
        self.channel_labels = self.model.channel_labels
        self.bar_labels = self.model.bar_labels

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

        # channel labels
        channel_labels = [
            torch.tensor([float(label.value) for label in self.channel_labels])
        ]

        # bar labels
        bar_labels = [torch.tensor([float(label.value) for label in self.bar_labels])]
        n_total_samples = sum(t.shape[0] * t.shape[2] for t in self.input_)
        bar_pad_zeros = n_total_samples - torch.numel(bar_labels[0])

        # pca outputs
        input_len = sum(torch.numel(t) for t in self.input_)
        pca_outputs_len = sum(torch.numel(t) for t in self.pca_outputs)
        pca_outputs_pad_zeros = input_len - pca_outputs_len

        electrodes_offset = torch.zeros(NUM_ELECTRODES)
        n_inputs = len(self.input_)
        for idx, input_ in enumerate(self.input_):
            electrodes_offset[idx + 1 : n_inputs + 1] += (
                torch.numel(input_) * FLOAT_SIZE_BYTES
            )

        mem = Mem()
        _ = mem.add_tensor_list_chunk([electrodes_offset], "electrodes_offset")
        _ = mem.add_tensor_list_chunk(
            spike_deeptector_params, "spike_deeptector_params"
        )
        _ = mem.add_tensor_list_chunk(bar_params, "bar_params")
        input_idx = mem.add_tensor_list_chunk(self.input_, "input")
        _ = mem.add_tensor_list_chunk([mem_0], "mem_0", masked=True)
        _ = mem.add_tensor_list_chunk([mem_1], "mem_1", masked=True)
        _ = mem.add_tensor_list_chunk(channel_labels, "channel_labels", masked=True)
        _ = mem.add_tensor_list_chunk(
            bar_labels, "bar_labels", pad_zeros=bar_pad_zeros, masked=True
        )
        _ = mem.add_tensor_list_chunk(
            self.pca_outputs, "output", pad_zeros=pca_outputs_pad_zeros, masked=True
        )

        # update electrodes offset with actual memory addresses
        electrodes_offset[: n_inputs + 1] += (
            mem.chunks[input_idx].offset * FLOAT_SIZE_BYTES
        )

        self.mem_obj = mem
        self.mem = mem.to_text()

        print(f"int mem_len = {mem.len};")

        for chunk in mem.chunks:
            if getattr(chunk, "name") is not None:
                print(f"int {chunk.name}_len = {chunk.len};")

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
