import typing as T

import torch
import numpy as np


class MemChunk:
    def __init__(
        self,
        data: T.List[torch.tensor],
        offset: int,
        name: T.Optional[str] = None,
        pad_zeros: T.Optional[int] = None,
        masked: bool = False,
        dtype: T.Optional[str] = None,
    ):
        """
        Create a memory chunk for the `Mem` object.

        - `data`: stored data
        - `offset`: address in the global memory map
        - `name`: (optional)
        - `pad_zeros`: add zeros at the end
        - `masked`: if `True`, then this chunk may be outputted as only zeros
        - `dtype`: data type for the output
        """
        self.data = data
        self.offset = offset
        self.name = name
        self.pad_zeros = pad_zeros
        self.masked = masked
        self.dtype = dtype

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

        np_data = [tensor.detach().numpy() for tensor in flat_data]

        if self.dtype is not None:
            np_data = [array.astype(self.dtype) for array in np_data]

        return [f"{x}\n" for array in flat_data for x in array]

    def to_bin(self, pre: bool = False) -> bytes:
        """
        Output a binary array with the values from the chunk.

        - `pre`: a flag to turn all data with `masked == True` into 0s.
        """
        mem_bin = b""

        flat_data = [torch.flatten(t) for t in self.data]

        if self.masked and pre:
            flat_data = [torch.zeros_like(t) for t in flat_data]

        for t in flat_data:
            array = torch.flatten(t).detach().numpy()
            if self.dtype is not None:
                array = array.astype(self.dtype)

            mem_bin += array.tobytes()

        if self.pad_zeros is not None:
            mem_bin += np.zeros(self.pad_zeros, dtype=array.dtype).tobytes()

        return mem_bin
