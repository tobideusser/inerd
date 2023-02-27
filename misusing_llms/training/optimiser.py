from importlib import import_module

import torch

OPTIMISERS = {
    "adam": "torch.optim.Adam",
    "adamW": "torch.optim.AdamW",  # torch.optim.AdamW | transformers.AdamW
    "adagrad": "torch.optim.Adagrad",
}


class Optimiser:
    @classmethod
    def from_config(cls, type_: str, *args, **kwargs) -> torch.optim.Optimizer:
        try:
            callable_path = OPTIMISERS[type_]
        except KeyError:
            raise KeyError(f"Optimiser '{type_}' is not implemented.")

        module_name, class_name = callable_path.rsplit(".", maxsplit=1)
        module = import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(*args, **kwargs)
