import logging
import os
from typing import Dict, List, Union

import torch
import pytorch_lightning as pl
import wandb
from fluidml import Task
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from pytorch_lightning.strategies import FSDPStrategy, DeepSpeedStrategy
from torch.utils.data import DataLoader, Dataset
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
        csv_logging: bool = False,
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
        self.csv_logging = csv_logging

        self.is_subprocess = "LOCAL_RANK" in os.environ

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
            num_workers = 25

        dataloaders = {}
        for split_type, split_dataset in datasets.items():

            dataloaders[split_type] = DataLoader(
                dataset=split_dataset,
                collate_fn=batch_collator,
                shuffle=True if split_type == "train" else False,
                num_workers=num_workers,
                # drop_last=True,
                **self.training_params["data_loading"],
            )

        return dataloaders

    def _init_model_loggers(self) -> Union[List, None]:
        store_context = self.get_store_context()
        if store_context and not self.is_subprocess:
            run_dir = store_context.run_dir
            run_id = self.id

            initialised_loggers = []

            if self.wandb_logging:
                initialised_loggers.append(WandbLogger(project=self.info.project_name, name=run_id, save_dir=run_dir))
                self._save_wandb_api_path()

            if self.tensorboard_logging:
                initialised_loggers.append(
                    TensorBoardLogger(save_dir=os.path.join(run_dir, "tensorboard"), name="", version="")
                )

            if self.csv_logging:
                initialised_loggers.append((CSVLogger(save_dir=run_dir, name="lightning_csv_logs")))

            if not (self.wandb_logging or self.tensorboard_logging or self.csv_logging):
                raise ValueError("Select at least one logger to allow tracking of the best epoch and model.")

            return initialised_loggers
        else:
            return None

    def _save_wandb_api_path(self):
        store_context = self.get_store_context()
        if store_context and not self.is_subprocess:
            run_dir = store_context.run_dir
            sub_dir = os.path.relpath(wandb.run.dir, run_dir)
            self.save(
                {"wandb_api_path": wandb.run.path},
                "wandb_api_path",
                type_="json",
                sub_dir=sub_dir,
            )

    def _log_hyperparameter(self, loggers):
        # hardcoded which hyperparameter will be logged
        if not self.is_subprocess:
            hyperparameter_to_be_logged = {
                "informed_generation": self.informed_generation,
                "batch_size": self.training_params["data_loading"]["batch_size"],
                "combine_train_valid": self.combine_train_valid,
                "model_name": self.model_params["model_name"],
                "n-bit precision": self.training_params["trainer"]["precision"],
                "strategy": "fsdp" if len(self.resource.device) > 1 else "",
                "num_gpus": len(self.resource.device),
            }
            for train_logger in loggers:
                if isinstance(train_logger, pl.loggers.tensorboard.TensorBoardLogger):
                    pass  # todo
                elif isinstance(train_logger, pl.loggers.wandb.WandbLogger):
                    train_logger.experiment.config.update(hyperparameter_to_be_logged)

    def _init_model_callbacks(self) -> List:
        callbacks = [
            ProgressBar(),
            LearningRateMonitor(logging_interval="step"),
            ExceptionHandling(),
        ]

        store_context = self.get_store_context()
        if store_context:
            run_dir = store_context.run_dir

            # if not self.is_subprocess:
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
            callbacks.append(model_checkpoint)

        if self.training_params["callbacks"].apply_early_stopping:
            if not self.is_subprocess:
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

        # this disables the warning that appears when using bloom (and others?)
        # see here:
        #   https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
        if "bloom" in self.model_params["model_name"]:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokeniser = AutoTokenizer.from_pretrained(self.model_params["model_name"], use_fast=False)

        batch_collator = NERBatchCollator(pad_token_id=tokeniser.pad_token_id)
        datasets = self._init_torch_datasets(corpus=corpus)
        dataloaders = self._init_torch_dataloaders(datasets, batch_collator)
        loggers = self._init_model_loggers()
        callbacks = self._init_model_callbacks()

        if self.training_params.get("metrics", False):
            evaluator = Evaluator.from_config(entity_set=corpus.entity_set, **self.training_params["metrics"])
        else:
            evaluator = None
        # entity_type_token_ids = tokeniser(text=sorted(list(corpus.entity_set)), add_special_tokens=False).input_ids

        if self.informed_generation:
            logits_processor = LogitsProcessorList()
            combine_token = (
                self.unique_config["Tokenisation"].get("combine_token", "\n")
                if "Tokenisation" in self.unique_config
                else "\n"
            )
            entity_type_tokens = sorted(list(corpus.entity_set))

            vocab_size = tokeniser.vocab_size
            if "bloom" in tokeniser.name_or_path:
                logger.debug("'Bloom' tokeniser chosen, adding 200 to vocab size for logits processor.")
                logger.debug("See: https://huggingface.co/bigscience/bloom-560m/discussions/43")
                vocab_size += 200

            logits_processor.append(
                InformedNERDecoderLogitsProcessor(
                    entity_type_tokens=entity_type_tokens,
                    vocab_size=vocab_size,
                    tokeniser=tokeniser,
                    combine_token=combine_token,
                    entity_separator_token=";",
                    type_content_separator_token=":",
                )
            )
        else:
            logits_processor = None

        if self.resource.cuda:
            accelerator = "gpu"
            gpus = self.resource.device
            if isinstance(gpus, list) and len(gpus) > 1:
                strategy = FSDPStrategy(cpu_offload=True)
                # strategy = DeepSpeedStrategy(
                #     stage=3,
                #     offload_optimizer=True,
                #     offload_parameters=True,
                # )
            else:
                strategy = "auto"
        else:
            accelerator = "cpu"
            gpus = None
            strategy = "auto"

        trainer = pl.Trainer(
            num_sanity_val_steps=0,
            accelerator=accelerator,
            devices=gpus,
            logger=loggers,
            callbacks=callbacks,
            strategy=strategy,
            plugins=FluidmlCheckpointIO(task=self),
            **self.training_params["trainer"],
        )

        if loggers is not None:
            self._log_hyperparameter(loggers=loggers)

        if self.training_params.get("lr_scheduler", False):
            total_devices = trainer.num_devices * trainer.num_nodes
            train_batches = len(dataloaders["train"]) // total_devices
            train_steps = (trainer.max_epochs * train_batches) // trainer.accumulate_grad_batches
            lr_warmup = self.training_params["lr_scheduler"].pop("lr_warmup", 0.0)
            interval = self.training_params["lr_scheduler"].pop("interval", "epoch")
            learning_rate_scheduler_inputs = self.training_params["lr_scheduler"]
            learning_rate_scheduler_inputs.update(
                {"num_warmup_steps": lr_warmup * train_steps, "num_training_steps": train_steps, "interval": interval}
            )
        else:
            learning_rate_scheduler_inputs = None

        model = GenerativeNERModel(
            model_params=self.model_params,
            optimiser_params=self.training_params["optimiser"],
            learning_rate_scheduler_inputs=learning_rate_scheduler_inputs,
            # evaluator_params=self.training_params["metrics"],
            evaluator=evaluator,
            tokeniser=tokeniser,
            logits_processor=logits_processor,
            is_multigpu=True if strategy != "auto" else False,
            is_mainprocess=not self.is_subprocess,
            entity_set=corpus.entity_set,
            # do_logging=self.is_subprocess,
        )

        trainer.fit(
            model=model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["validation"],
            # train_dataloaders=dataloader,
            # val_dataloaders=dataloader,
            ckpt_path="last" if self.warm_start else None,
        )

        if not self.is_subprocess:
            wandb.finish()
