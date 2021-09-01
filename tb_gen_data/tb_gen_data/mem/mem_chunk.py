import typing as T
from array import array

import torch


class MemChunk:
    def __init__(
        self,
        data: T.List[torch.tensor],
        offset: int,
        name: T.Optional[str] = None,
        pad_zeros: T.Optional[int] = None,
        masked: bool = False,
    ):
        """
        Create a memory chunk for the `Mem` object.

        - `data`: stored data
        - `offset`: address in the global memory map
        - `name`: (optional)
        - `pad_zeros`: add zeros at the end
        - `masked`: if `True`, then this chunk may be outputted as only zeros
        """
        self.data = data
        self.offset = offset
        self.name = name
        self.pad_zeros = pad_zeros
        self.masked = masked

        self.len = sum(torch.numel(t) for t in data)
        if pad_zeros is not None:
            self.len += pad_zeros

    def to_txt(self, pre: bool = False) -> T.List[str]:
        """
        Output a list of strings with the values from the chunk.

        - `pre`: a flag to turn all data with `masked == True` into 0s.
        """
        flat_data = [torch.flatten(t) for t in self.data]

        if self.masked and pre:
            flat_data = [torch.zeros_like(t) for t in flat_data]

        if self.pad_zeros is not None:
            flat_data += [torch.zeros(self.pad_zeros)]

        return [f"{x}\n" for tensor in flat_data for x in tensor.tolist()]

    def to_bin(self, pre: bool = False) -> array:
        """
        Output a binary array with the values from the chunk.

        - `pre`: a flag to turn all data with `masked == True` into 0s.
        """
        mem_bin = array("f")

        flat_data = [torch.flatten(t) for t in self.data]

        if self.masked and pre:
            flat_data = [torch.zeros_like(t) for t in flat_data]

        for t in flat_data:
            mem_bin.fromlist(torch.flatten(t).tolist())

        if self.pad_zeros is not None:
            mem_bin.fromlist([0.0] * self.pad_zeros)

        return mem_bin
