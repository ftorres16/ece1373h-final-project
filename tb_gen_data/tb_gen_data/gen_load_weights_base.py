import typing as T

import torch
from torch import nn

from config import OUTPUT_FOLDER
from tb_gen_data.gen_base import GenBase
from tb_gen_data.models.zero_mean import ZeroMean


class GenLoadWeightsBase(GenBase):
    """
    Same as GenBase, but with extra functions for loading weights from files.
    """

    def __init__(
        self, name: str, mem_weights_file: str, out_folder: str = OUTPUT_FOLDER
    ):
        super().__init__(name, out_folder)

        self.mem_weights_file = mem_weights_file

    def _load_mem_file(
        self, mem_weights_file: T.Optional[str] = None, model: nn.Module = None
    ):
        if mem_weights_file is None:
            mem_weights_file = self.mem_weights_file

        if model is None:
            model = self.model

        with open(f"{self.out_folder}/{mem_weights_file}") as f:
            mem = [float(line.strip()) for line in f.readlines()]

        from_idx = 0
        to_idx = 0
        for layer in model.layers:
            from_idx = to_idx

            if isinstance(layer, ZeroMean):
                to_idx = self._load_mem_entry(mem, from_idx, layer, "mean")
            elif isinstance(layer, nn.Conv2d):
                to_idx = self._load_mem_entry(mem, from_idx, layer, "weight")
                to_idx = self._load_mem_entry(mem, to_idx, layer, "bias")
            elif isinstance(layer, nn.BatchNorm2d):
                to_idx = self._load_mem_1d(
                    mem, from_idx, layer, "running_mean", n=layer.num_features
                )
                to_idx = self._load_mem_1d(
                    mem, to_idx, layer, "running_var", n=layer.num_features
                )
                to_idx = self._load_mem_1d(
                    mem, to_idx, layer, "weight", n=layer.num_features
                )
                to_idx = self._load_mem_1d(
                    mem, to_idx, layer, "bias", n=layer.num_features
                )

        in_len = 1 * 1 * self.in_y * self.in_x
        from_idx = to_idx
        to_idx = from_idx + in_len
        # self.input_ = torch.tensor(mem[from_idx:to_idx]).reshape(
        #     (1, 1, self.in_y, self.in_x)
        # )

    def _load_mem_entry(
        self,
        mem: T.List[float],
        from_idx: int,
        layer: nn.Module,
        param: str,
    ) -> int:
        """
        Helper for `_load_mem_file`. Load values from `mem[from_idx]` info `layer.param`
        """
        to_idx = from_idx + len(torch.flatten(getattr(layer, param)))

        param_tensor = mem[from_idx:to_idx]
        param_tensor = torch.tensor(param_tensor)
        param_tensor = param_tensor.reshape(getattr(layer, param).shape)
        param_tensor = nn.Parameter(param_tensor)
        param_tensor.requires_grad = False

        setattr(layer, param, param_tensor)

        return to_idx

    def _load_mem_1d(
        self, mem: T.List[float], from_idx: int, layer: nn.Module, param: str, n: int
    ) -> int:
        """
        Helper for `_load_mem_file`. Load values from `mem[from_idx]` info `layer.param` in 1d.
        """
        to_idx = from_idx + n

        param_tensor = torch.tensor(mem[from_idx:to_idx])
        param_tensor = nn.Parameter(param_tensor)
        param_tensor.requires_grad = False

        setattr(layer, param, param_tensor)

        return to_idx

    def _get_model_params(
        self, model: T.Optional[nn.Module] = None
    ) -> T.List[torch.tensor]:
        """
        Get all model params as a list.
        """
        if model is None:
            model = self.model

        params = []

        for idx, layer in enumerate(model.layers):
            if isinstance(layer, ZeroMean):
                params.append(layer.mean)
            elif isinstance(layer, nn.Conv2d):
                params.append(layer.weight)
                params.append(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                params.append(layer.running_mean)
                params.append(layer.running_var)
                params.append(
                    layer.weight
                    if layer.weight is not None
                    else torch.tensor([1.0] * layer.num_features)
                )
                params.append(
                    layer.bias
                    if layer.bias is not None
                    else torch.tensor([0.0] * layer.num_features)
                )

        return params
