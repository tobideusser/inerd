import json
import logging
import os
from datetime import timedelta
from typing import Dict, List, Union, Optional

import pytorch_lightning as pl
import wandb
from fluidml import Task
from huggingface_hub import login as hf_login
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import FSDPStrategy, DeepSpeedStrategy
from transformers import AutoTokenizer, LogitsProcessorList, LlamaTokenizer

from inerd import project_path
from inerd.data_classes import NERCorpus
from inerd.models import GenerativeNERModel, GenerativeNERModelFSDP, GenerativeNERModelDeepSpeed
from inerd.training import (
    NERBatchCollator,
    GenerativeNERDataset,
    FluidmlCheckpointIO,
    InformedNERDecoderLogitsProcessor,
)
from inerd.training.callbacks import init_model_callbacks
from inerd.training.dataloader import init_torch_dataloaders
from inerd.utils import set_seeds
from inerd.utils.fluid_helper import log_to_file

logger = logging.getLogger(__name__)


class NERTraining(Task):
    def __init__(
        self,
        training_params: Dict,
        model_params: Dict,
        generation_params: Dict,
        dataset: str,
        gpu_scaling: str = "auto",
        seed: int = 3141,
        warm_start: bool = False,
        wandb_logging: bool = False,
        csv_logging: bool = True,
        pre_training: bool = True,
        no_zero_shot: bool = False,
        checkpointing_time_interval: Optional[float] = None,
    ):
        super().__init__()

        self.training_params = training_params
        self.model_params = model_params
        self.informed_generation = generation_params.pop("informed_generation")
        self.generation_params = generation_params
        self.seed = seed
        self.dataset_name = dataset
        self.gpu_scaling = gpu_scaling
        self.warm_start = warm_start
        self.zero_shot = not no_zero_shot
        self.checkpointing_time_interval = (
            timedelta(seconds=checkpointing_time_interval) if checkpointing_time_interval is not None else None
        )

        self.combine_train_valid = self.training_params["data_loading"].pop("combine_train_valid", False)

        self.pre_training = pre_training

        self.wandb_logging = wandb_logging
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

    def _init_model_loggers(self) -> Union[List, None]:
        store_context = self.get_store_context()
        if store_context and not self.is_subprocess:
            run_dir = store_context.run_dir
            run_id = self.id

            initialised_loggers = []

            if self.wandb_logging:
                if self.warm_start:
                    try:
                        path_to_api_key = os.path.join(
                            store_context.run_dir, "wandb", "latest-run", "files", "wandb_api_path.json"
                        )
                        with open(path_to_api_key) as file:
                            wandb_api_path = json.load(file)
                        wandb_id = wandb_api_path["wandb_api_path"].split("/")[-1]
                    except FileNotFoundError:
                        wandb_id = None
                else:
                    wandb_id = None
                if self.training_params["trainer"]["max_epochs"] > 1:
                    project_name = self.info.project_name + "finetuning"
                else:
                    project_name = self.info.project_name + "onefewshot"
                initialised_loggers.append(
                    WandbLogger(project=project_name, name=run_id, save_dir=run_dir, id=wandb_id)
                )
                self._save_wandb_api_path()

            if self.csv_logging:
                initialised_loggers.append((CSVLogger(save_dir=run_dir, name="lightning_csv_logs")))

            if not (self.wandb_logging or self.csv_logging):
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
            if self.model_params["llama"]:
                if "7B" in self.model_params["model_name"]:
                    model_name = "llama-7B"
                elif "13B" in self.model_params["model_name"]:
                    model_name = "llama-13B"
                else:
                    model_name = "llama-?B"
            else:
                model_name = self.model_params["model_name"]
            logger.info(f"Using model {model_name}.")
            hyperparameter_to_be_logged = {
                "informed_generation": self.informed_generation,
                "batch_size": self.training_params["data_loading"]["batch_size"],
                "combine_train_valid": self.combine_train_valid,
                "model_name": model_name,
                "n-bit precision": self.training_params["trainer"]["precision"],
                "strategy": "fsdp" if len(self.resource.device) > 1 else "",
                "num_gpus": len(self.resource.device),
                "model_8bit": self.model_params["load_in_8bit"],
                "lora": self.model_params["lora"],
                "accumulate_grad_batches": self.training_params["trainer"]["accumulate_grad_batches"],
                "effective_batch_size": self.training_params["data_loading"]["batch_size"]
                * self.training_params["trainer"]["accumulate_grad_batches"],
                "max_epochs": self.training_params["trainer"]["max_epochs"],
                "early_stopping": "true_patience=" + str(self.training_params["callbacks"]["patience"])
                if self.training_params["callbacks"]["apply_early_stopping"]
                else "false",
                "dataset": self.dataset_name,
                "pre_training": self.pre_training,
                "max_new_tokens": self.generation_params["max_new_tokens"],
            }
            for train_logger in loggers:
                if isinstance(train_logger, pl.loggers.wandb.WandbLogger):
                    train_logger.experiment.config.update(hyperparameter_to_be_logged)

    @log_to_file
    def run(self, corpus_tokenised: NERCorpus, best_model: Optional[Dict] = None):
        if isinstance(corpus_tokenised, Dict):
            logger.info("Converting corpus_tokenised dict to NERCorpus object.")
            corpus = NERCorpus.from_dict(corpus_tokenised)
            entity_separator_token = corpus[0].entity_separator_token
            type_content_separator_token = corpus[0].type_content_separator_token
        elif isinstance(corpus_tokenised, list):
            logger.info(
                f"Converting corpus_parsed list of dict to a NERCorpus object which only holds the {self.dataset_name} "
                f"dataset."
            )
            corpus = [
                NERCorpus.from_dict(c)
                for c in corpus_tokenised
                if c["name"].replace("-", "") == self.dataset_name.replace("-", "")
            ][0]
            entity_separator_token = corpus[0].entity_separator_token
            type_content_separator_token = corpus[0].type_content_separator_token
        else:
            corpus = corpus_tokenised
            entity_separator_token = corpus[0].entity_separator_token
            type_content_separator_token = corpus[0].type_content_separator_token
        max_len = 700
        logger.info(f"Setting max input length to {max_len}.")
        i = 0
        to_delete = []
        for ii, sentence in enumerate(corpus.train):
            if len(sentence.input_tokens) > max_len:
                to_delete.append(ii)
        for index in sorted(to_delete, reverse=True):
            del corpus.train[index]
            i += 1
        logger.info(f"Deleted {i} occurences exceeding the max input length of {max_len}.")
        set_seeds(self.seed)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if self.model_params["llama"]:
            tokeniser = LlamaTokenizer.from_pretrained(self.model_params["model_name"])
        elif "Llama-2" in self.model_params["model_name"]:
            path_to_token = os.path.join(project_path, "hf_token.txt")
            with open(path_to_token, "r") as file:
                hf_token = file.read().rstrip()
            logger.info(f"Login to HuggingFace with the token stored under {path_to_token}")
            hf_login(token=hf_token)
            tokeniser = AutoTokenizer.from_pretrained(self.model_params["model_name"], token=hf_token)
        else:
            tokeniser = AutoTokenizer.from_pretrained(self.model_params["model_name"])

        if entity_separator_token not in tokeniser.get_vocab():
            tokeniser.add_special_tokens(
                {
                    "additional_special_tokens": [entity_separator_token],
                }
            )
        if type_content_separator_token not in tokeniser.get_vocab():
            tokeniser.add_special_tokens(
                {
                    "additional_special_tokens": [type_content_separator_token],
                }
            )

        if self.model_params["llama"]:
            tokeniser.add_special_tokens({"pad_token": "<PAD>"})
            pad_token_id = tokeniser.pad_token_id
        elif "RedPajama" in self.model_params["model_name"]:
            pad_token_id = 1  # "<|padding|>" in GPT-NEOX
            tokeniser.pad_token_id = 1
        elif (
            "falcon" in self.model_params["model_name"]
            or "gpt2" in self.model_params["model_name"]
            or "stanford-crfm/BioMedLM" in self.model_params["model_name"]
        ):
            tokeniser.add_special_tokens({"pad_token": "<|padding|>"})
            pad_token_id = tokeniser.pad_token_id
        elif tokeniser.pad_token_id is None:
            raise NotImplementedError
        else:
            pad_token_id = tokeniser.pad_token_id

        batch_collator = NERBatchCollator(pad_token_id=pad_token_id)
        datasets = self._init_torch_datasets(corpus=corpus)
        dataloaders = init_torch_dataloaders(
            datasets=datasets, batch_collator=batch_collator, logger=logger, **self.training_params["data_loading"]
        )
        loggers = self._init_model_loggers()
        if self.get_store_context():
            callbacks = init_model_callbacks(
                run_dir=self.get_store_context().run_dir,
                gpu_scaling=self.gpu_scaling,
                monitor_var=self.training_params["callbacks"].monitor_var,
                save_top_k=self.training_params["callbacks"].save_top_k,
                monitor_var_mode=self.training_params["callbacks"].monitor_var_mode,
                checkpointing_time_interval=self.checkpointing_time_interval,
                apply_early_stopping=self.training_params["callbacks"].apply_early_stopping,
                is_subprocess=self.is_subprocess,
                patience=self.training_params["callbacks"].patience,
            )
        else:
            callbacks = init_model_callbacks()

        vocab_size = len(tokeniser)
        if self.informed_generation:
            logits_processor = LogitsProcessorList()
            combine_token = (
                self.unique_config["Tokenisation"].get("combine_token", "\n")
                if "Tokenisation" in self.unique_config
                else "\n"
            )
            entity_type_tokens = sorted(list(corpus.entity_set))

            if "RedPajama" in tokeniser.name_or_path:
                logger.debug("'RedPajama' tokeniser chosen, fixing vocab_size to 50432.")
                vocab_size = 50432

            logits_processor.append(
                InformedNERDecoderLogitsProcessor(
                    entity_type_tokens=entity_type_tokens,
                    vocab_size=vocab_size,
                    tokeniser=tokeniser,
                    combine_token=combine_token,
                    entity_separator_token=entity_separator_token,
                    type_content_separator_token=type_content_separator_token,
                    batch_size=self.training_params["data_loading"]["batch_size"],
                    leading_space=not self.model_params["llama"],
                )
            )
        else:
            logits_processor = None

        if self.resource.cuda:
            accelerator = "gpu"
            gpus = self.resource.device
            if isinstance(gpus, list) and len(gpus) > 1:
                if self.gpu_scaling in ["fsdp", "auto"]:
                    strategy = FSDPStrategy(cpu_offload=False)
                elif self.gpu_scaling == "deepspeed":
                    strategy = DeepSpeedStrategy(
                        stage=3,
                        offload_optimizer=True,
                        offload_parameters=True,
                    )
                else:
                    strategy = "auto"
            else:
                strategy = "auto"
        else:
            accelerator = "cpu"
            gpus = "auto"
            strategy = "auto"

        trainer = pl.Trainer(
            num_sanity_val_steps=0,
            accelerator=accelerator,
            devices=gpus,
            logger=loggers,
            callbacks=callbacks,
            strategy=strategy,
            plugins=FluidmlCheckpointIO(task=self),
            log_every_n_steps=1,
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

        init_model_parameter = {
            "model_params": self.model_params,
            "optimiser_params": self.training_params["optimiser"],
            "generation_params": self.generation_params,
            "learning_rate_scheduler_inputs": learning_rate_scheduler_inputs,
            "tokeniser": tokeniser,
            "logits_processor": logits_processor,
            "is_multigpu": True if strategy != "auto" else False,
            "is_mainprocess": not self.is_subprocess,
            "entity_set": corpus.entity_set,
            "pad_token_id": pad_token_id,
            "type_content_separator_token": type_content_separator_token,
            "entity_separator_token": entity_separator_token,
            "do_zero_shot": True,
            "hf_cache_dir": os.path.join(self.results_store.base_dir, ".hfcache"),
            "vocab_size": vocab_size,
        }

        if strategy == "auto":
            model = GenerativeNERModel(**init_model_parameter)
        elif isinstance(strategy, FSDPStrategy):
            model = GenerativeNERModelFSDP(**init_model_parameter)
        elif isinstance(strategy, DeepSpeedStrategy):
            model = GenerativeNERModelDeepSpeed(**init_model_parameter)
        else:
            raise ValueError()

        if not self.warm_start:
            if self.pre_training:
                model.load_state_dict(best_model["state_dict"])

            if self.zero_shot:
                logger.info("Doing zero-shot evaluation on test set.")
                zero_shot_metrics = trainer.test(model=model, dataloaders=dataloaders["test"])
                logger.info("Zero-shot metrics:")
                logger.info(zero_shot_metrics)

        trainer.fit(
            model=model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["validation"],
            ckpt_path="last" if self.warm_start else None,
        )

        if not self.is_subprocess:
            wandb.finish()
