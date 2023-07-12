import json
import logging
import os
from datetime import timedelta
from typing import Dict, List, Union, Optional

import pytorch_lightning as pl
import wandb
from fluidml import Task
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from pytorch_lightning.strategies import FSDPStrategy, DeepSpeedStrategy
from transformers import AutoTokenizer, LlamaTokenizer

from inerd.data_classes import NERCorpus
from inerd.models import GenerativeNERModel, GenerativeNERModelFSDP, GenerativeNERModelDeepSpeed
from inerd.training import (
    NERBatchCollator,
    GenerativeNERDataset,
    FluidmlCheckpointIO,
)
from inerd.training.callbacks import init_model_callbacks
from inerd.training.dataloader import init_torch_dataloaders
from inerd.utils import set_seeds
from inerd.utils.fluid_helper import log_to_file

logger = logging.getLogger(__name__)


class NERPreTraining(Task):
    def __init__(
        self,
        training_params: Dict,
        model_params: Dict,
        generation_params: Dict,
        gpu_scaling: str = "auto",
        seed: int = 3141,
        warm_start: bool = False,
        wandb_logging: bool = False,
        csv_logging: bool = True,
        checkpointing_time_interval: Optional[float] = None,
    ):
        super().__init__()

        self.training_params = training_params
        self.model_params = model_params
        self.informed_generation = generation_params.pop("informed_generation")
        self.generation_params = generation_params
        self.seed = seed
        self.warm_start = warm_start
        self.checkpointing_time_interval = (
            timedelta(seconds=checkpointing_time_interval) if checkpointing_time_interval is not None else None
        )
        self.gpu_scaling = gpu_scaling

        self.combine_train_valid = self.training_params["data_loading"].pop("combine_train_valid", False)

        self.wandb_logging = wandb_logging
        self.csv_logging = csv_logging

        self.is_subprocess = "LOCAL_RANK" in os.environ

    def _init_torch_datasets(self, corpus: List[NERCorpus]) -> Dict[str, GenerativeNERDataset]:
        test_sentences = []
        valid_sentences = []
        train_sentences = []
        for c in corpus:
            test_sentences.extend(c.test)
            valid_sentences.extend(c.validation)
            train_sentences.extend(c.train)

        datasets = {"test": GenerativeNERDataset(sentences=test_sentences)}

        if self.combine_train_valid:
            datasets["train"] = GenerativeNERDataset(sentences=train_sentences + valid_sentences)
            logger.info(
                f"Combining training and validation set for a total training length of {len(datasets['train'])} "
                f"sentences."
            )
        else:
            datasets["train"] = GenerativeNERDataset(sentences=train_sentences)
            logger.info(f"Training length: {len(datasets['train'])} sentences.")

        if self.training_params["data_loading"].pop("validate_on_test_set", False):
            datasets["validation"] = GenerativeNERDataset(sentences=test_sentences)
            logger.info(f"Using test as validation set with length: {len(datasets['validation'])} sentences.")
            logger.warning("Test set and validation are now the same, beware of this!")
        else:
            datasets["validation"] = GenerativeNERDataset(sentences=valid_sentences)
            logger.info(f"Validation length: {len(datasets['validation'])} sentences.")

        logger.info(f"Test length: {len(datasets['test'])} sentences.")

        return datasets

    # def _init_torch_dataloaders(
    #     self,
    #     datasets: Dict[str, GenerativeNERDataset],
    #     batch_collator: NERBatchCollator,
    # ) -> Dict[str, DataLoader]:
    #
    #     if is_debug():
    #         logger.warning(
    #             "Debug mode detected, setting num_workers=0 for torch dataloader. This allows proper debugging."
    #         )
    #         num_workers = 0
    #     else:
    #         num_workers = 25
    #
    #     dataloaders = {}
    #     for split_type, split_dataset in datasets.items():
    #
    #         dataloaders[split_type] = DataLoader(
    #             dataset=split_dataset,
    #             collate_fn=batch_collator,
    #             shuffle=True if split_type == "train" else False,
    #             num_workers=num_workers,
    #             # drop_last=True,
    #             **self.training_params["data_loading"],
    #         )
    #
    #     return dataloaders

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
                initialised_loggers.append(
                    WandbLogger(
                        project=self.info.project_name + "pretraining", name=run_id, save_dir=run_dir, id=wandb_id
                    )
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
            datasets = self.unique_config["Parsing"]["dataset"]
            hyperparameter_to_be_logged = {
                "informed_generation": self.informed_generation,
                "batch_size": self.training_params["data_loading"]["batch_size"],
                "combine_train_valid": self.combine_train_valid,
                "model_name": model_name,
                "n-bit precision": self.training_params["trainer"]["precision"],
                "strategy": self.gpu_scaling,
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
            }
            hyperparameter_to_be_logged.update(datasets)
            for train_logger in loggers:
                if isinstance(train_logger, pl.loggers.wandb.WandbLogger):
                    train_logger.experiment.config.update(hyperparameter_to_be_logged)

    # def _init_model_callbacks(self) -> List:
    #     callbacks = [
    #         ProgressBar(),
    #         LearningRateMonitor(logging_interval="step"),
    #         ExceptionHandling(),
    #     ]
    #
    #     store_context = self.get_store_context()
    #     if store_context:
    #         run_dir = store_context.run_dir
    #
    #         # if not self.is_subprocess:
    #
    #         if self.gpu_scaling == "deepspeed":
    #             model_checkpoint = ModelCheckpoint(
    #                 monitor="epoch",
    #                 every_n_epochs=1,
    #                 dirpath=os.path.join(run_dir, "models"),
    #                 verbose=True,
    #                 save_last=True,
    #                 save_on_train_epoch_end=True,
    #                 save_top_k=-1,
    #                 save_weights_only=True,
    #             )
    #         else:
    #             model_checkpoint = ModelCheckpoint(
    #                 monitor=self.training_params["callbacks"].monitor_var,
    #                 dirpath=os.path.join(run_dir, "models"),
    #                 filename="best_model",
    #                 save_top_k=self.training_params["callbacks"].save_top_k,
    #                 verbose=True,
    #                 save_last=True,
    #                 mode=self.training_params["callbacks"].monitor_var_mode,
    #             )
    #         model_checkpoint.FILE_EXTENSION = ""  # handled by fluidml file store
    #         callbacks.append(model_checkpoint)
    #
    #         if self.checkpointing_time_interval:
    #             model_checkpoint_time = ModelCheckpoint(
    #                 monitor=self.training_params["callbacks"].monitor_var,
    #                 dirpath=os.path.join(run_dir, "models"),
    #                 filename="time_ckpt",
    #                 save_top_k=1,
    #                 verbose=True,
    #                 save_last=True,
    #                 mode=self.training_params["callbacks"].monitor_var_mode,
    #                 train_time_interval=self.checkpointing_time_interval,
    #             )
    #             model_checkpoint_time.FILE_EXTENSION = ""
    #             callbacks.append(model_checkpoint_time)
    #
    #     if self.training_params["callbacks"].apply_early_stopping:
    #         if not self.is_subprocess:
    #             callbacks.append(
    #                 EarlyStopping(
    #                     monitor=self.training_params["callbacks"].monitor_var,
    #                     mode=self.training_params["callbacks"].monitor_var_mode,
    #                     patience=self.training_params["callbacks"].patience,
    #                 )
    #             )
    #
    #     return callbacks

    @log_to_file
    def run(self, corpus_tokenised: NERCorpus):

        if isinstance(corpus_tokenised, Dict):
            logger.info("Converting corpus_tokenised dict to NERCorpus object.")
            corpus = NERCorpus.from_dict(corpus_tokenised)
            entity_separator_token = corpus[0].entity_separator_token
            type_content_separator_token = corpus[0].type_content_separator_token
        elif isinstance(corpus_tokenised, list):
            logger.info("Converting corpus_parsed list of dict to list of NERCorpus object.")
            corpus = [NERCorpus.from_dict(c) for c in corpus_tokenised]
            entity_separator_token = corpus[0][0].entity_separator_token
            type_content_separator_token = corpus[0][0].type_content_separator_token
            max_len = 700
            logger.info(f"Setting max input length to {max_len}.")
            i = 0
            for c in corpus:
                to_delete = []
                for ii, sentence in enumerate(c.train):
                    if len(sentence.input_tokens) > max_len:
                        to_delete.append(ii)
                for index in sorted(to_delete, reverse=True):
                    del c.train[index]
                    i += 1
            logger.info(f"Deleted {i} occurences exceeding the max input length of {max_len}.")
            # a = []
            # for c in corpus:
            #     for i, sentence in enumerate(c.train):
            #         a.append(len(sentence.input_tokens))
            # print(a)
        else:
            corpus = corpus_tokenised
            entity_separator_token = corpus[0].entity_separator_token
            type_content_separator_token = corpus[0].type_content_separator_token

        set_seeds(self.seed)

        # this disables the warning that appears when using bloom, opt, and RedPajama (and others?)
        # see here:
        #   https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
        if (
            "bloom" in self.model_params["model_name"]
            or "RedPajama" in self.model_params["model_name"]
            or "opt" in self.model_params["model_name"]
            or "gpt-2" in self.model_params["model_name"]
            or "falcon" in self.model_params["model_name"]
            or self.model_params["llama"]
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if self.model_params["llama"]:
            tokeniser = LlamaTokenizer.from_pretrained(self.model_params["model_name"])
        else:
            tokeniser = AutoTokenizer.from_pretrained(self.model_params["model_name"])

        tokeniser.add_special_tokens(
            {
                "additional_special_tokens": [entity_separator_token, type_content_separator_token],
            }
        )

        if self.model_params["llama"]:
            tokeniser.add_special_tokens({"pad_token": "<PAD>"})
            pad_token_id = tokeniser.pad_token_id
        elif "RedPajama" in self.model_params["model_name"]:
            pad_token_id = 1  # "<|padding|>" in GPT-NEOX
            tokeniser.pad_token_id = 1
        elif "falcon" in self.model_params["model_name"] or "gpt2" in self.model_params["model_name"]:
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
        # dataloaders = self._init_torch_dataloaders(datasets, batch_collator)
        loggers = self._init_model_loggers()
        # callbacks = self._init_model_callbacks()
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

        entity_set = set()
        for c in corpus:
            entity_set.update(c.entity_set)

        # if self.informed_generation:
        #     logits_processor = LogitsProcessorList()
        #     combine_token = (
        #         self.unique_config["Tokenisation"].get("combine_token", "\n")
        #         if "Tokenisation" in self.unique_config
        #         else "\n"
        #     )
        #     entity_type_tokens = sorted(list(corpus.entity_set))
        #
        #     vocab_size = tokeniser.vocab_size
        #     if "bloom" in tokeniser.name_or_path:
        #         logger.debug("'Bloom' tokeniser chosen, adding 200 to vocab size for logits processor.")
        #         logger.debug("See: https://huggingface.co/bigscience/bloom-560m/discussions/43")
        #         vocab_size += 200
        #     elif "RedPajama" in tokeniser.name_or_path:
        #         logger.debug("'RedPajama' tokeniser chosen, adding 178 to vocab size for logits processor.")
        #         vocab_size += 178
        #     elif self.model_params["llama"] or "falcon" in tokeniser.name_or_path:
        #         vocab_size = len(tokeniser)
        #
        #     logits_processor.append(
        #         InformedNERDecoderLogitsProcessor(
        #             entity_type_tokens=entity_type_tokens,
        #             vocab_size=vocab_size,
        #             tokeniser=tokeniser,
        #             combine_token=combine_token,
        #             entity_separator_token=entity_separator_token,
        #             type_content_separator_token=type_content_separator_token,
        #             batch_size=self.training_params["data_loading"]["batch_size"],
        #             leading_space=not self.model_params["llama"],
        #         )
        #     )
        # else:
        #     logits_processor = None

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
                # if "bloom" in self.model_params["model_name"]:
                #     strategy = FSDPStrategy(cpu_offload=True, activation_checkpointing=BloomBlock)
                # elif "opt" in self.model_params["model_name"]:
                #     strategy = FSDPStrategy(cpu_offload=True, activation_checkpointing=OPTDecoderLayer)
                # else:
                #     strategy = FSDPStrategy(cpu_offload=False)
                # strategy = DeepSpeedStrategy(
                #     stage=3,
                #     offload_optimizer=True,
                #     offload_parameters=True,
                # )
            else:
                strategy = "auto"
        else:
            accelerator = "cpu"
            gpus = "auto"
            strategy = "auto"

        # max_epochs = self.training_params["trainer"].pop("max_epochs")

        trainer = pl.Trainer(
            num_sanity_val_steps=0,
            limit_val_batches=0 if self.gpu_scaling == "deepspeed" else None,
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

        init_model_parameter = {
            "model_params": self.model_params,
            "optimiser_params": self.training_params["optimiser"],
            "generation_params": self.generation_params,
            "learning_rate_scheduler_inputs": learning_rate_scheduler_inputs,
            "tokeniser": tokeniser,
            # logits_processor=logits_processor,
            "is_multigpu": True if strategy != "auto" else False,
            "is_mainprocess": not self.is_subprocess,
            "entity_set": entity_set,
            "pad_token_id": pad_token_id,
            "type_content_separator_token": type_content_separator_token,
            "entity_separator_token": entity_separator_token,
            "hf_cache_dir": os.path.join(self.results_store.base_dir, ".hfcache"),
        }

        if strategy == "auto":
            model = GenerativeNERModel(**init_model_parameter)
        elif isinstance(strategy, FSDPStrategy):
            model = GenerativeNERModelFSDP(**init_model_parameter)
        elif isinstance(strategy, DeepSpeedStrategy):
            model = GenerativeNERModelDeepSpeed(**init_model_parameter)
        else:
            raise ValueError()

        trainer.fit(
            model=model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["validation"],
            ckpt_path="last" if self.warm_start else None,
        )

        if not self.is_subprocess:
            wandb.finish()
