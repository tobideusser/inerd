import logging
from typing import Dict, Any

import markdown
import pytorch_lightning as pl
import wandb
from fluidml import Task
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)


def log_text(key: str, log_string: str, trainer: pl.Trainer) -> None:
    for train_logger in trainer.loggers:
        if not log_string.startswith("\t"):
            log_string = fix_formatting(log_string)
        if isinstance(train_logger, pl.loggers.TensorBoardLogger):
            log_text_tensorboard(key, log_string, trainer, train_logger)
        elif isinstance(train_logger, pl.loggers.WandbLogger):
            log_text_wandb(key, log_string, trainer)
        else:
            logger.error(f"pl.Trainer.logger of type {type(train_logger)} can not store text.")


def fix_formatting(log_string: str) -> str:
    """
    In markdown, we create a code block by indenting each line with a tab \t.
    Code blocks have fixed formatting, i.e. each character has the same width and
    spacing is kept.
    This makes sure that texts like classification reports are printed as they appear
    in the console.
    Parameters
    ----------
    log_string
    Returns
    -------
    formatted log_string
    """
    lines = log_string.split("\n")
    lines = ["\t" + line for line in lines]
    return "\n".join(lines)


def log_text_tensorboard(key: str, log_string: str, trainer: pl.Trainer, train_logger: TensorBoardLogger):
    train_logger.experiment.add_text(key, log_string, global_step=trainer.current_epoch)


def log_text_wandb(key: str, log_string: str, trainer: pl.Trainer):
    try:
        wandb.log(
            {
                key: wandb.Html(markdown_to_html(log_string)),
                "epoch": trainer.current_epoch,
                "batch": trainer.global_step,
            }
        )
    except wandb.errors.Error as e:
        logger.warning(
            f"Unable to log string with wandb. "
            f"If this happens in the validation sanity check, you can ignore this message. "
            f'Error: "{str(e)}"'
        )


def markdown_to_html(markdown_string: str) -> str:
    return markdown.markdown(markdown_string)


class ExceptionHandling(Callback):
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        # re-raise the KeyboardInterrupt if caught (currently not raised by pytorch lightning)
        if isinstance(exception, KeyboardInterrupt):
            raise exception


class ProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
