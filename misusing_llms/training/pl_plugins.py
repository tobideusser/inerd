import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from fluidml import Task
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO

log = logging.getLogger(__name__)


class FluidmlCheckpointIO(CheckpointIO):
    """CheckpointIO that utilizes fluidml file store to save, load and remove checkpoints."""

    def __init__(self, task: Task):
        self.task = task

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: Union[str, Path],
        storage_options: Optional[Any] = None,
    ) -> None:

        sub_dir, name = os.path.split(path)
        sub_dir = sub_dir if sub_dir else None
        self.task.save(checkpoint, name=name, sub_dir=sub_dir, type_="pl_checkpoint")

    def load_checkpoint(
        self,
        path: Union[str, Path],
        map_location: Optional[Callable] = lambda storage, loc: storage,
    ) -> Dict[str, Any]:
        """Loads checkpoint using :func:`torch.load`, with additional handling for ``fsspec`` remote loading of
        files.

        Args:
            path: Path to checkpoint
            map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
            locations.

        Returns: The loaded checkpoint.

        Raises:
            FileNotFoundError: If ``path`` is not found by the ``fsspec`` filesystem
        """
        name = Path(path).stem
        return self.task.load(
            name=name, type_="pl_checkpoint", map_location=map_location
        )

    def remove_checkpoint(self, path: Union[str, Path]) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint
        """
        name = Path(path).stem
        self.task.delete(name=name)
