import logging
import os
from typing import Dict, List

import pytorch_lightning as pl
from fluidml import Task
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from misusing_llms.data_classes import NERCorpus
from misusing_llms.models import GenerativeNERModel
from misusing_llms.training import (
    NERBatchCollator,
    GenerativeNERDataset,
    ProgressBar,
    ExceptionHandling,
    FluidmlCheckpointIO,
)
from misusing_llms.utils.fluid_helper import log_to_file
from misusing_llms.utils.utils import set_seeds, set_device

logger = logging.getLogger(__name__)


class NERTraining(Task):
    def __init__(
        self,
        training_params: Dict,
        model_params: Dict,
        seed: int = 42,
        warm_start: bool = False,
        wandb_logging: bool = True,
        tensorboard_logging: bool = False,
    ):
        super().__init__()

        self.training_params = training_params
        self.model_params = model_params
        self.seed = seed
        self.warm_start = warm_start

        self.wandb_logging = wandb_logging
        self.tensorboard_logging = tensorboard_logging

    def _init_torch_datasets(self, corpus: NERCorpus) -> Dict[str, GenerativeNERDataset]:
        datasets = {"test": GenerativeNERDataset(sentences=corpus.test)}

        if self.training_params["data_loading"].pop("combine_train_valid", False):
            datasets["train"] = GenerativeNERDataset(sentences=corpus.train + corpus.validation)
            logger.info(
                f"Combining training and validation set for a total training length of {len(datasets['train'])} "
                f"sentences."
            )
        else:
            datasets["train"] = GenerativeNERDataset(sentences=corpus.train)
            logger.info(f"Training length: {len(datasets['train'])} sentences.")

        if self.training_params["data_loading"].pop("validate_on_test_set", False):
            datasets["validation"] = GenerativeNERDataset(sentences=corpus.test)
            logger.info(f"Using test as validation set with length: {len(datasets['validation'])} sentences.")
            logger.warning("Test set and validation are now the same, beware of this!")
        else:
            datasets["validation"] = GenerativeNERDataset(sentences=corpus.validation)
            logger.info(f"Validation length: {len(datasets['validation'])} sentences.")

        logger.info(f"Test length: {len(datasets['test'])} sentences.")

        return datasets

    def _init_torch_dataloaders(
        self,
        datasets: Dict[str, GenerativeNERDataset],
        batch_collator: NERBatchCollator,
    ) -> Dict[str, DataLoader]:

        dataloaders = {}
        for split_type, split_dataset in datasets.items():

            dataloaders[split_type] = DataLoader(
                dataset=split_dataset,
                collate_fn=batch_collator,
                shuffle=True if split_type == "train" else False,
                **self.training_params["data_loading"],
            )

        return dataloaders

    def _init_model_loggers(self) -> List:
        run_dir = self.get_store_context()
        run_name = os.path.split(run_dir)[-1]

        initialised_loggers = []

        if self.wandb_logging:
            # todo: run_info.run_name will be changed in the final 0.3 fluidml release
            initialised_loggers.append(WandbLogger(project=self.run_info.project_name, name=run_name, save_dir=run_dir))
            self._save_wandb_api_path()

        if self.tensorboard_logging:
            initialised_loggers.append(
                TensorBoardLogger(save_dir=os.path.join(run_dir, "tensorboard"), name="", version="")
            )

        return initialised_loggers

    def _save_wandb_api_path(self):
        import wandb

        run_dir = self.get_store_context()
        sub_dir = os.path.relpath(wandb.run.dir, run_dir)
        self.save(
            {"wandb_api_path": wandb.run.path},
            "wandb_api_path",
            type_="json",
            sub_dir=sub_dir,
        )

    def _init_model_callbacks(self) -> List:
        run_dir = self.get_store_context()

        model_checkpoint = ModelCheckpoint(
            monitor=self.training_params["callbacks"].monitor_var,
            dirpath=os.path.join(run_dir, "models"),
            filename="best_model",
            save_top_k=self.training_params["callbacks"].save_top_k,
            verbose=True,
            save_last=True,
            mode=self.training_params["callbacks"].monitor_var_mode,
        )
        model_checkpoint.FILE_EXTENSION = ""  # handled by fluidml file store

        callbacks = [
            ProgressBar(),
            LearningRateMonitor(logging_interval="step"),
            model_checkpoint,
            ExceptionHandling(),
        ]

        if self.training_params["callbacks"].apply_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=self.training_params["callbacks"].monitor_var,
                    mode=self.training_params["callbacks"].monitor_var_mode,
                    patience=self.training_params["callbacks"].patience,
                )
            )

        return callbacks

    @log_to_file
    def run(self, corpus_tokenised: NERCorpus, tokeniser: PreTrainedTokenizer):
        if isinstance(corpus_tokenised, Dict):
            logger.info("Converting corpus_tokenised dict to Corpus object.")
            corpus = NERCorpus.from_dict(corpus_tokenised)
        else:
            corpus = corpus_tokenised

        set_seeds(self.seed)
        device = self.resource.device
        set_device(device)

        batch_collator = NERBatchCollator(pad_token_id=tokeniser.pad_token_id)
        datasets = self._init_torch_datasets(corpus=corpus)
        dataloaders = self._init_torch_dataloaders(datasets, batch_collator)
        loggers = self._init_model_loggers()
        callbacks = self._init_model_callbacks()

        model = GenerativeNERModel(
            model_params=self.model_params,
            optimiser_params=self.training_params["optimiser"],
            lr_scheduler_params=self.training_params.get("lr_scheduler", None),
            evaluator_params=self.training_params["metrics"],
        ).to(device)

        try:
            gpus = [int(device.split(":")[-1])]
            accelerator = "gpu"
        except ValueError:
            gpus = None
            accelerator = "cpu"

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=gpus,
            logger=loggers,
            callbacks=callbacks,
            plugins=FluidmlCheckpointIO(task=self),
            **self.training_params["trainer"],
        )

        trainer.fit(
            model=model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["validation"],
            ckpt_path="last" if self.warm_start else None,
        )
