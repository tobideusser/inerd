from importlib import import_module
from typing import Union

import deepspeed
import torch


OPTIMISERS = {
    "adam": "torch.optim.Adam",
    "adam8bit": "bitsandbytes.optim.Adam8bit",
    "adamW": "torch.optim.AdamW",  # torch.optim.AdamW | transformers.AdamW
    "adamW8bit": "bitsandbytes.optim.AdamW8bit",
    "adagrad": "torch.optim.Adagrad",
    "FusedAdam": "deepspeed.ops.adam.FusedAdam",
    "DeepSpeedCPUAdam": "deepspeed.ops.adam.DeepSpeedCPUAdam",
}


class Optimiser:
    @classmethod
    def from_config(
        cls, type_: str, multigpu: bool = False, load_in_8bit: bool = False, *args, **kwargs
    ) -> Union[torch.optim.Optimizer, deepspeed.ops.adam.FusedAdam, deepspeed.ops.adam.DeepSpeedCPUAdam]:
        if load_in_8bit:
            type_ = type_ + "8bit"

        multigpu = False  # DEBUG!
        if multigpu:
            model_params = kwargs.pop("params")
            if type_.lower() == "adamw":
                adamw_mode = True
            elif type_.lower() == "adam":
                adamw_mode = False
            else:
                raise KeyError("Optimiser must be 'adam' or 'adamw' to work with DeepSpeedCPUAdam")
            return deepspeed.ops.adam.DeepSpeedCPUAdam(model_params=model_params, adamw_mode=adamw_mode, **kwargs)
        else:
            try:
                callable_path = OPTIMISERS[type_]
            except KeyError:
                raise KeyError(f"Optimiser '{type_}' is not implemented.")

            module_name, class_name = callable_path.rsplit(".", maxsplit=1)
            module = import_module(module_name)
            class_ = getattr(module, class_name)
            return class_(*args, **kwargs)
