import functools
import io
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Dict, Callable, Any, List

import fsspec
import torch
from fluidml.storage import LocalFileStore, TypeInfo, StoreContext
from lightning_fabric.plugins.io import TorchCheckpointIO
from lightning_fabric.utilities.cloud_io import _load as pl_load
from lightning_fabric.utilities.cloud_io import get_filesystem
from rich.logging import RichHandler
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def atomic_save(checkpoint: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

    Copied from pytorch lightning, as functionality was deprecated in v1.8.0 and will be removed in v2.0.0!

    Args:
        checkpoint: The object to save.
            Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
            accepts.
        filepath: The path to which the checkpoint will be saved.
            This points to the file that the checkpoint will be stored in.
    """
    bytesbuffer = io.BytesIO()
    torch.save(checkpoint, bytesbuffer)
    with fsspec.open(filepath, "wb") as f:
        f.write(bytesbuffer.getvalue())


class MyLocalFileStore(LocalFileStore):
    def __init__(self, base_dir: str):
        super().__init__(base_dir=base_dir)

        self.type_registry["torch"] = TypeInfo(torch.save, torch.load, "pt", is_binary=True)
        self.type_registry["tokeniser"] = TypeInfo(self._save_tokeniser, self._load_tokeniser, needs_path=True)
        self.type_registry["pl_checkpoint"] = TypeInfo(
            self._save_pl_checkpoint,
            self._load_pl_checkpoint,
            "ckpt",
            is_binary=True,
            needs_path=True,
        )

        self.torch_checkpoint_io = TorchCheckpointIO()

    @staticmethod
    def _save_tokeniser(obj: PreTrainedTokenizerFast, path: str):
        obj.save_pretrained(save_directory=path, legacy_format=False)

    @staticmethod
    def _load_tokeniser(path: str) -> PreTrainedTokenizerFast:
        return PreTrainedTokenizerFast.from_pretrained(path)

    def _save_pl_checkpoint(self, checkpoint: Dict[str, Any], path: Union[str, Path]):
        self.torch_checkpoint_io.save_checkpoint(checkpoint=checkpoint, path=path, storage_options=None)
        # fs = get_filesystem(path)
        # fs.makedirs(os.path.dirname(path), exist_ok=True)
        # atomic_save(checkpoint, path)

    def _load_pl_checkpoint(
        self,
        path: Union[str, Path],
        map_location: Optional[Callable] = lambda storage, loc: storage,
    ) -> Dict[str, Any]:
        return self.torch_checkpoint_io.load_checkpoint(path=path, map_location=map_location)
        # Try to read the checkpoint at `path`. If not exist, do not restore checkpoint.
        # fs = get_filesystem(path)
        # if not fs.exists(path):
        #     raise FileNotFoundError(f"Checkpoint at {path} not found. Aborting training.")
        # return pl_load(path, map_location=map_location)


@dataclass
class TaskResource:
    cuda: bool
    device: Optional[Union[int, List[int], str]] = None


def configure_logging(level: Union[str, int] = "INFO", log_dir: Optional[str] = None):
    assert level in ["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "FATAL", "CRITICAL", 10, 20, 30, 40, 50]
    logger = logging.getLogger()
    formatter = logging.Formatter("%(processName)-13s%(message)s")
    stream_handler = RichHandler(rich_tracebacks=True, tracebacks_extra_lines=2, show_path=False)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    if log_dir is not None:
        log_path = os.path.join(log_dir, f"{datetime.now()}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("%(processName)s - %(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.addHandler(stream_handler)
    logger.setLevel(level)


def add_file_handler(log_dir: str, name: str = "logs", type_: str = "txt", level: Union[str, int] = "INFO"):
    if level not in ["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "FATAL", "CRITICAL", 10, 20, 30, 40, 50]:
        raise ValueError(f'Logging level "{level}" is not supported.')

    log_path = os.path.join(log_dir, f"{name}.{type_}")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(processName)s - %(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger_ = logging.getLogger()
    logger_.addHandler(file_handler)


def remove_file_handler():
    logger_ = logging.getLogger()
    logger_.handlers = [h for h in logger_.handlers if not isinstance(h, logging.FileHandler)]


def log_to_file(func):
    """Decorator to enable file logging for fluid ml tasks"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        store_context = self.get_store_context()
        if store_context:
            logger.info(f"Current run dir: {store_context.run_dir}")
            add_file_handler(store_context.run_dir)
            result = func(self, *args, **kwargs)
            remove_file_handler()
        else:
            result = func(self, *args, **kwargs)
        return result

    return wrapper
