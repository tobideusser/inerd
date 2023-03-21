import logging
import os
from typing import Dict, List

import pytorch_lightning as pl
import wandb
from fluidml import Task
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LogitsProcessorList

from misusing_llms.data_classes import NERCorpus
from misusing_llms.models import GenerativeNERModel
from misusing_llms.training import (
    NERBatchCollator,
    GenerativeNERDataset,
    ProgressBar,
    ExceptionHandling,
    FluidmlCheckpointIO,
    Evaluator,
    InformedNERDecoderLogitsProcessor,
)
from misusing_llms.utils.fluid_helper import log_to_file
from misusing_llms.utils import set_seeds, set_device, is_debug

logger = logging.getLogger(__name__)


class NERTraining(Task):
    def __init__(
        self,
        training_params: Dict,
        model_params: Dict,
        informed_generation: bool = False,
        seed: int = 3141,
        warm_start: bool = False,
        wandb_logging: bool = True,
        tensorboard_logging: bool = False,
    ):
        super().__init__()

        self.training_params = training_params
        self.model_params = model_params
        self.informed_generation = informed_generation
        self.seed = seed
        self.warm_start = warm_start

        self.combine_train_valid = self.training_params["data_loading"].pop("combine_train_valid", False)

        self.wandb_logging = wandb_logging
        self.tensorboard_logging = tensorboard_logging

    def _init_torch_datasets(self, corpus: NERCorpus) -> Dict[str, GenerativeNERDataset]:
        datasets = {"test": GenerativeNERDataset(sentences=corpus.test)}

        if self.combine_train_valid:
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

        if is_debug():
            logger.warning(
                "Debug mode detected, setting num_workers=0 for torch dataloader. This allows proper debugging."
            )
            num_workers = 0
        else:
            num_workers = 10

        dataloaders = {}
        for split_type, split_dataset in datasets.items():

            dataloaders[split_type] = DataLoader(
                dataset=split_dataset,
                collate_fn=batch_collator,
                shuffle=True if split_type == "train" else False,
                num_workers=num_workers,
                **self.training_params["data_loading"],
            )

        return dataloaders

    def _init_model_loggers(self) -> List:
        run_dir = self.get_store_context().run_dir
        run_id = self.id

        initialised_loggers = []

        if self.wandb_logging:
            initialised_loggers.append(WandbLogger(project=self.info.project_name, name=run_id, save_dir=run_dir))
            self._save_wandb_api_path()

        if self.tensorboard_logging:
            initialised_loggers.append(
                TensorBoardLogger(save_dir=os.path.join(run_dir, "tensorboard"), name="", version="")
            )

        return initialised_loggers

    def _save_wandb_api_path(self):
        run_dir = self.get_store_context().run_dir
        sub_dir = os.path.relpath(wandb.run.dir, run_dir)
        self.save(
            {"wandb_api_path": wandb.run.path},
            "wandb_api_path",
            type_="json",
            sub_dir=sub_dir,
        )

    def _log_hyperparameter(self, loggers):
        # hardcoded which hyperparameter will be logged
        hyperparameter_to_be_logged = {
            "informed_generation": self.informed_generation,
            "batch_size": self.training_params["data_loading"]["batch_size"],
            "combine_train_valid": self.combine_train_valid,
            "model_name": self.model_params["model_name"],
        }
        for train_logger in loggers:
            if isinstance(train_logger, pl.loggers.tensorboard.TensorBoardLogger):
                pass  # todo
            elif isinstance(train_logger, pl.loggers.wandb.WandbLogger):
                train_logger.experiment.config.update(hyperparameter_to_be_logged)

    def _init_model_callbacks(self) -> List:
        run_dir = self.get_store_context().run_dir

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
    def run(self, corpus_tokenised: NERCorpus):
        if isinstance(corpus_tokenised, Dict):
            logger.info("Converting corpus_tokenised dict to Corpus object.")
            corpus = NERCorpus.from_dict(corpus_tokenised)
        else:
            corpus = corpus_tokenised

        set_seeds(self.seed)
        device = self.resource.device
        set_device(device)

        tokeniser = AutoTokenizer.from_pretrained(self.model_params["model_name"], use_fast=True)

        batch_collator = NERBatchCollator(pad_token_id=tokeniser.pad_token_id)
        datasets = self._init_torch_datasets(corpus=corpus)
        dataloaders = self._init_torch_dataloaders(datasets, batch_collator)
        loggers = self._init_model_loggers()
        callbacks = self._init_model_callbacks()

        if self.training_params.get("metrics", False):
            evaluator = Evaluator.from_config(entity_set=corpus.entity_set, **self.training_params["metrics"])
        else:
            evaluator = None
        entity_type_token_ids = tokeniser(text=sorted(list(corpus.entity_set)), add_special_tokens=False).input_ids
        if self.informed_generation:
            logits_processor = LogitsProcessorList()
            logits_processor.append(
                InformedNERDecoderLogitsProcessor(entity_type_token_ids=entity_type_token_ids, t=tokeniser)
            )
        else:
            logits_processor = None

        model = GenerativeNERModel(
            model_params=self.model_params,
            optimiser_params=self.training_params["optimiser"],
            lr_scheduler_params=self.training_params.get("lr_scheduler", None),
            # evaluator_params=self.training_params["metrics"],
            evaluator=evaluator,
            tokeniser=tokeniser,
            logits_processor=logits_processor,
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

        self._log_hyperparameter(loggers=loggers)

        trainer.fit(
            model=model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["validation"],
            ckpt_path="last" if self.warm_start else None,
        )

        wandb.finish()
