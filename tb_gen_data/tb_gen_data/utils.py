import typing as T

import torch


def gen_mem_overwrite(src: T.List[torch.tensor]) -> T.Tuple[torch.tensor]:
    """
    Give the output memoty in two memory tensors, each created by
    overwriting the previous input with the next output.
    """
    mem_0 = []
    mem_1 = []

    for idx, tensor in enumerate(src):
        if idx % 2 == 0:
            tgt = mem_0
        else:
            tgt = mem_1

        flat_tensor = torch.flatten(tensor)
        tgt[: len(flat_tensor)] = flat_tensor

    return torch.flatten(torch.tensor(mem_0)), torch.flatten(torch.tensor(mem_1))
