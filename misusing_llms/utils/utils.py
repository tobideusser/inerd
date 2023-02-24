import multiprocessing
import os
import random
from typing import List, Optional, Union, Dict

import numpy as np
import torch

_SEED: Optional[int] = None
_DEVICE: Optional[torch.device] = None


def get_balanced_devices(
    count: Optional[int] = None, use_cuda: bool = True, cuda_ids: Optional[List[int]] = None
) -> List[str]:
    count = count if count is not None else multiprocessing.cpu_count()
    if use_cuda and torch.cuda.is_available():
        if cuda_ids is not None:
            devices = [f"cuda:{id_}" for id_ in cuda_ids]
        else:
            devices = [f"cuda:{id_}" for id_ in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
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


def get_padding_mask(seqlens: List[int], device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """

    :param seqlens:
    :type seqlens: List[int]
    :param device:
    :type device: Union[str, torch.device]
    :return: A tensor (with boolean values) that indicates which tokens in the s
    :rtype: torch.Tensor
    """

    mask = torch.zeros((len(seqlens), max(seqlens)), dtype=torch.bool, device=device)
    for i, l in enumerate(seqlens):
        mask[i, :l] = 1
    return mask


def iob2pred(batch, vocab):
    """Convert iob predictions into list of entities ({'start', 'end', 'type_'})"""
    # todo: merge iob2pred and extract_iob
    pred_iobes_ids = batch["ner_output"]
    pred_iobes = [[vocab.iobes.idx2val[iobes_id] for iobes_id in sample.cpu().numpy()] for sample in pred_iobes_ids]
    pred_iobes = [p[:n_words] for p, n_words in zip(pred_iobes, batch["n_words"])]

    return [extract_iob(p) for p in pred_iobes]


def extract_iob(iob_tags):
    """Convert list of IOB tags into list of entities ({'start', 'end', 'type'})"""
    entities = {}

    tmp_indices = None
    tmp_type = "O"
    for i, t in enumerate(iob_tags):
        if t[0] == "B":
            if tmp_indices is not None:
                entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
                # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})
            tmp_type = "-".join(t.split("-")[1:])
            tmp_indices = [i]

        elif t[0] == "O":
            if tmp_indices is not None:
                entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
                # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})
            tmp_type = None
            tmp_indices = None

        elif t[0] == "I":
            if "-".join(t.split("-")[1:]) == tmp_type and i == tmp_indices[-1] + 1:
                tmp_indices += [i]
            else:
                if tmp_indices is not None:
                    entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
                    # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})
                tmp_type = "-".join(t.split("-")[1:])
                tmp_indices = [i]

    if tmp_indices is not None:
        entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
        # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})

    return entities


def gini(x: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculates Gini coefficient (see Gini, Corrado. "Variabilità e mutabilità (Variability and Mutability)." or the
    entry on wikipedia: https://en.wikipedia.org/wiki/Gini_coefficient). From wikipedia: The Gini coefficient measures
    the inequality among values of a frequency distribution, such as the levels of income. A Gini coefficient of 0
    reflects perfect equality, where all income or wealth values are the same, while a Gini coefficient of 1 (or 100%)
    reflects maximal inequality among values.


    Code as been adapted from: https://stackoverflow.com/a/49571213


    :param x: the vector for which the Gini coefficient should be calculated
    :type x: either a numpy array or torch tensor
    :return: the Gini coefficient (a value between 0 and 1) of the vector x
    :rtype: float
    """
    # convert torch tensor to np array
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    # check if x is a vector
    if x.ndim != 1:
        raise ValueError("x must a 1-D vector.")

    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)

    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def normalise_to_zero_one_interval(x: torch.Tensor):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def bound_at_zero(x: torch.Tensor):
    x_min = x.min()
    if x_min < 0:
        return x - x_min
    else:
        return x


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
