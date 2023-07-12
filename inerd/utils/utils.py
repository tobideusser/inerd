import multiprocessing
import operator
import os
import random
import sys
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch

from inerd.data_classes import Entity

_SEED: Optional[int] = None
_DEVICE: Optional[torch.device] = None


def rindex(lst: List, value: Any):
    """returns the last occurence of value in lst"""
    return len(lst) - operator.indexOf(reversed(lst), value) - 1


def entity_string_to_entity_dataclass(
    entity_string: str, entity_separator_token: str, type_content_separator_token: str
) -> Union[List[Entity], List]:
    if entity_separator_token in entity_string:
        entity_blocks = entity_string.split(entity_separator_token)
        entities = []
        for entity_block in entity_blocks:
            if type_content_separator_token in entity_block:
                split = entity_block.split(type_content_separator_token)
                if len(split[0]) > 0 and len(split[1]) > 0:
                    # remove leading and trailing space if it exists
                    entity_type = split[0].strip()
                    entity_words = split[1].strip()
                    # entity_type = split[0] if split[0][0] != " " else split[0][1:]
                    # entity_words = split[1] if split[1][0] != " " else split[1][1:]
                    entities.append(Entity(words=entity_words, type_=entity_type))
        return entities
    else:
        return []


def is_debug():
    gettrace = getattr(sys, "gettrace", None)

    if gettrace is None:
        return False
    else:
        v = gettrace()
        if v is None:
            return False
        else:
            return True


def get_balanced_devices(
    count: Optional[int] = None,
    use_cuda: bool = True,
    cuda_ids: Optional[List[int]] = None,
    cuda_group: Optional[int] = None,
) -> Union[List[str], List[List[int]]]:
    if use_cuda and torch.cuda.is_available():
        if cuda_group is not None and cuda_group > 1:
            if not use_cuda:
                raise ValueError(f"cuda_groups is set to {cuda_group}, but use_cuda is set to false.")
            if cuda_ids is not None:
                if (len(cuda_ids) / cuda_group).is_integer():
                    devices = [cuda_ids[x : x + cuda_group] for x in range(0, len(cuda_ids), cuda_group)]
                    # devices = [[f"cuda:{id_}" for id_ in id_group] for id_group in cuda_ids_grouped]
                else:
                    raise ValueError(
                        f"cuda_groups is set to {cuda_group}, but cuda_ids is of length {len(cuda_ids)} and thus not "
                        f"divisible by {cuda_group}."
                    )
            else:
                raise NotImplementedError
                # pass  # todo here! -> devices = [f"cuda:{id_}" for id_ in range(torch.cuda.device_count())]
        else:
            if cuda_ids is not None:
                devices = [[cuda_id] for cuda_id in cuda_ids]
            else:
                devices = [[id_] for id_ in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    count = count if count is not None else multiprocessing.cpu_count()
    factor = int(count / len(devices))
    remainder = count % len(devices)
    devices = devices * factor + devices[:remainder]
    return devices


def set_seed_number(seed: int):
    global _SEED
    _SEED = seed


def set_seeds(seed: int = 3141):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_device(device: str):
    global _DEVICE
    _DEVICE = torch.device(device)


def get_device() -> torch.device:
    return _DEVICE


def get_devices_for_pl(only_ids: bool = False) -> Union[List, Dict]:
    try:
        gpus = [int(str(_DEVICE).split(":")[-1])]
        accelerator = "gpu"
    except ValueError:
        gpus = None
        accelerator = "cpu"
    if only_ids:
        return gpus
    else:
        return {"devices": gpus, "accelerator": accelerator}
