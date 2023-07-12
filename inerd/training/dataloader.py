import logging
from typing import Optional, Dict

from torch.utils.data import DataLoader

from inerd.training import GenerativeNERDataset, NERBatchCollator
from inerd.utils import is_debug


def init_torch_dataloaders(
    datasets: Dict[str, GenerativeNERDataset],
    batch_collator: NERBatchCollator,
    logger: Optional[logging.Logger] = None,
    **dataloading_kwargs,
) -> Dict[str, DataLoader]:
    if is_debug():
        logger.warning("Debug mode detected, setting num_workers=0 for torch dataloader. This allows proper debugging.")
        num_workers = 0
    else:
        num_workers = 25

    dataloaders = {}
    for split_type, split_dataset in datasets.items():
        dataloaders[split_type] = DataLoader(
            dataset=split_dataset,
            collate_fn=batch_collator,
            shuffle=True if split_type == "train" else False,
            num_workers=num_workers,
            **dataloading_kwargs,
        )

    return dataloaders
