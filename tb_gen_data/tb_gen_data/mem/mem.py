import typing as T
from array import array

import torch
import numpy as np

from tb_gen_data.mem.mem_chunk import MemChunk

FLOAT_SIZE_BYTES = 4
BYTE_SIZE_BITS = 8
BIT_ALIGNMENT = 512


class Mem:
    def __init__(self, dtype: T.Optional[str] = None):
        self.len = 0
        self.chunks = []
        self.dtype = dtype

    def add_tensor_list_chunk(
        self,
        data: T.List[torch.tensor],
        name: T.Optional[str] = None,
        pad_zeros: T.Optional[int] = None,
        masked: bool = False,
    ) -> int:
        """
        Only works correctly if all chunks are the same data type. Returns chunk index.

        - pad_zeros: number of zeros to append at the end.
        - masked: when creating an output with `pre == True`, this chunk should be all zeros.
        """
        chunk = MemChunk(
            data,
            offset=self.len,
            name=name,
            pad_zeros=pad_zeros,
            masked=masked,
            dtype=self.dtype,
        )

        self.chunks.append(chunk)

        self.len += chunk.len

        return len(self.chunks) - 1

    def to_text(self, pre: bool = False) -> T.List[str]:
        return [entry for chunk in self.chunks for entry in chunk.to_txt(pre=pre)]

    def to_bin(self, pre: bool = False) -> bytes:
        mem_bin = b""

        for chunk in self.chunks:
            mem_bin += chunk.to_bin(pre=pre)

        # memory needs alignment to BIT_ALIGNMENT bits
        mem_bin_size = len(mem_bin) * BYTE_SIZE_BITS
        pad_bits = (
            BIT_ALIGNMENT - mem_bin_size % BIT_ALIGNMENT
            if mem_bin_size % BIT_ALIGNMENT != 0
            else 0
        )

        if pad_bits % BYTE_SIZE_BITS != 0:
            print(
                f"Warning! Padding bits {pad_bits} can't be realized with 8 bit bytes."
            )

        pad_bytes = pad_bits // BYTE_SIZE_BITS

        mem_bin += np.zeros(pad_bytes, dtype="byte").tobytes()

        return mem_bin
