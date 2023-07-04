import os
from datetime import timedelta
from typing import List, Optional

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from misusing_llms.training import ProgressBar, ExceptionHandling


def init_model_callbacks(
    run_dir: Optional[str] = None,
    gpu_scaling: str = "auto",
    monitor_var: Optional[str] = None,
    save_top_k: Optional[int] = None,
    monitor_var_mode: Optional[str] = None,
    checkpointing_time_interval: Optional[timedelta] = None,
    apply_early_stopping: Optional[bool] = None,
    is_subprocess: bool = False,
    patience: Optional[int] = None,
) -> List:
    callbacks = [
        ProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        ExceptionHandling(),
    ]
    if run_dir:

        if monitor_var is None and gpu_scaling != "deepspeed":
            raise ValueError("No metric to monitor specified. Set the function parameter monitor_var.")
        if save_top_k is None and gpu_scaling != "deepspeed":
            save_top_k = 1
        if monitor_var_mode is None and gpu_scaling != "deepspeed":
            raise ValueError(
                f"No mode specified for the metric {monitor_var}. Set the function parameter monitor_var_mode to 'min' "
                f"or 'max'"
            )

        if gpu_scaling == "deepspeed":
            model_checkpoint = ModelCheckpoint(
                monitor="epoch",
                every_n_epochs=1,
                dirpath=os.path.join(run_dir, "models"),
                verbose=True,
                save_last=True,
                save_on_train_epoch_end=True,
                save_top_k=-1,
                save_weights_only=True,
            )
        else:
            model_checkpoint = ModelCheckpoint(
                monitor=monitor_var,
                dirpath=os.path.join(run_dir, "models"),
                filename="best_model",
                save_top_k=save_top_k,
                verbose=True,
                save_last=True,
                mode=monitor_var_mode,
            )
        model_checkpoint.FILE_EXTENSION = ""  # handled by fluidml file store
        callbacks.append(model_checkpoint)

        if checkpointing_time_interval is not None:
            model_checkpoint_time = ModelCheckpoint(
                monitor=monitor_var,
                dirpath=os.path.join(run_dir, "models"),
                filename="time_ckpt",
                save_top_k=1,
                verbose=True,
                save_last=True,
                mode=monitor_var_mode,
                train_time_interval=checkpointing_time_interval,
            )
            model_checkpoint_time.FILE_EXTENSION = ""
            callbacks.append(model_checkpoint_time)

        if apply_early_stopping and not is_subprocess:
            if patience is None:
                patience = 1
                UserWarning("No patience specified, setting it to 1.")
            callbacks.append(EarlyStopping(monitor=monitor_var, mode=monitor_var_mode, patience=patience))

    return callbacks
