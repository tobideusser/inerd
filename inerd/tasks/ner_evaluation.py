import logging
import os
from copy import deepcopy
from typing import Optional, Union, List, Dict

import pytorch_lightning as pl
from fluidml import Task
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import FSDPStrategy, DeepSpeedStrategy
from transformers import AutoTokenizer, LogitsProcessorList, LlamaTokenizer

from inerd.data_classes import NERCorpus
from inerd.models import GenerativeNERModel, GenerativeNERModelFSDP, GenerativeNERModelDeepSpeed
from inerd.training import (
    NERBatchCollator,
    GenerativeNERDataset,
    ProgressBar,
    FluidmlCheckpointIO,
    InformedNERDecoderLogitsProcessor,
)
from inerd.training.dataloader import init_torch_dataloaders

logger = logging.getLogger(__name__)


class NEREvaluation(Task):
    def __init__(self, split: Optional[Union[str, List]] = None):
        super().__init__()
        if split is None:
            split = "test"
        if isinstance(split, list):
            self.split = split
        else:
            self.split = [split]
        self.is_subprocess = "LOCAL_RANK" in os.environ

    def _init_torch_datasets(self, corpus: NERCorpus) -> Dict[str, GenerativeNERDataset]:
        datasets = {}
        if "test" in self.split:
            datasets["test"] = GenerativeNERDataset(sentences=corpus.test)
        if "validation" in self.split:
            datasets["validation"] = GenerativeNERDataset(sentences=corpus.validation)
        if "train" in self.split:
            datasets["train"] = GenerativeNERDataset(sentences=corpus.train)
        return datasets

    def _init_model_loggers(
        self, wandb_logging: bool = False, csv_logging: bool = True, onefewshot: bool = False
    ) -> Union[List, None]:
        store_context = self.get_store_context()
        if store_context:
            run_dir = store_context.run_dir
            run_id = self.id

            initialised_loggers = []

            if wandb_logging:
                wandb_api_path = self.load(name="wandb_api_path", task_name="NERTraining")
                wandb_id = wandb_api_path["wandb_api_path"].split("/")[-1]
                if onefewshot:
                    project_name = self.info.project_name + "onefewshot"
                else:
                    project_name = self.info.project_name + "finetuning"
                initialised_loggers.append(
                    WandbLogger(project=project_name, name=run_id, save_dir=run_dir, id=wandb_id)
                )

            if csv_logging:
                initialised_loggers.append((CSVLogger(save_dir=run_dir, name="lightning_csv_logs")))

            if not (wandb_logging or csv_logging):
                raise ValueError("Select at least one logger to allow tracking of the best epoch and model.")

            return initialised_loggers
        else:
            return None

    # @staticmethod
    # def _init_torch_dataloaders(
    #     datasets: Dict[str, GenerativeNERDataset],
    #     batch_collator: NERBatchCollator,
    #     training_params: Dict,
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
    #             shuffle=False,
    #             num_workers=num_workers,
    #             # drop_last=True,
    #             **training_params["data_loading"],
    #         )
    #
    #     return dataloaders

    def run(self, best_model: Dict, corpus_tokenised: NERCorpus):

        model_params = deepcopy(self.unique_config["NERTraining"]["model_params"])
        training_params = deepcopy(self.unique_config["NERTraining"]["training_params"])
        generation_params = deepcopy(self.unique_config["NERTraining"]["generation_params"])
        dataset_name = deepcopy(self.unique_config["NERTraining"]["dataset"])
        wandb_logging = deepcopy(self.unique_config["NERTraining"]["wandb_logging"])
        csv_logging = deepcopy(self.unique_config["NERTraining"]["csv_logging"])
        informed_generation = generation_params.pop("informed_generation")
        del training_params["data_loading"]["combine_train_valid"]
        del training_params["data_loading"]["validate_on_test_set"]

        if isinstance(corpus_tokenised, Dict):
            logger.info("Converting corpus_tokenised dict to NERCorpus object.")
            corpus = NERCorpus.from_dict(corpus_tokenised)
            entity_separator_token = corpus[0].entity_separator_token
            type_content_separator_token = corpus[0].type_content_separator_token
        elif isinstance(corpus_tokenised, list):
            logger.info("Converting corpus_parsed list of dict to list of NERCorpus object.")
            corpus = [
                NERCorpus.from_dict(c)
                for c in corpus_tokenised
                if c["name"].replace("-", "") == dataset_name.replace("-", "")
            ][0]
            entity_separator_token = corpus[0].entity_separator_token
            type_content_separator_token = corpus[0].type_content_separator_token
        else:
            corpus = corpus_tokenised
            entity_separator_token = corpus[0].entity_separator_token
            type_content_separator_token = corpus[0].type_content_separator_token

        if (
            "bloom" in model_params["model_name"]
            or "RedPajama" in model_params["model_name"]
            or "opt" in model_params["model_name"]
            or "gpt-2" in model_params["model_name"]
            or model_params["llama"]
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if model_params["llama"]:
            tokeniser = LlamaTokenizer.from_pretrained(model_params["model_name"])
        else:
            tokeniser = AutoTokenizer.from_pretrained(model_params["model_name"])

        tokeniser.add_special_tokens(
            {
                "additional_special_tokens": [entity_separator_token, type_content_separator_token],
            }
        )

        if model_params["llama"]:
            tokeniser.add_special_tokens({"pad_token": "<|padding|>"})
            pad_token_id = tokeniser.pad_token_id
        elif "RedPajama" in model_params["model_name"]:
            pad_token_id = 1  # "<|padding|>" in GPT-NEOX
            tokeniser.pad_token_id = 1
        elif "falcon" in model_params["model_name"] or "gpt2" in model_params["model_name"]:
            tokeniser.add_special_tokens({"pad_token": "<|padding|>"})
            pad_token_id = tokeniser.pad_token_id
        elif tokeniser.pad_token_id is None:
            raise NotImplementedError
        else:
            pad_token_id = tokeniser.pad_token_id

        batch_collator = NERBatchCollator(pad_token_id=pad_token_id)
        datasets = self._init_torch_datasets(corpus=corpus)
        dataloaders = init_torch_dataloaders(
            datasets=datasets, batch_collator=batch_collator, logger=logger, **training_params["data_loading"]
        )
        loggers = self._init_model_loggers(
            wandb_logging=wandb_logging,
            csv_logging=csv_logging,
            onefewshot=training_params["trainer"]["max_epochs"] <= 1,
        )
        # dataloaders = self._init_torch_dataloaders(
        #     datasets=datasets, batch_collator=batch_collator, training_params=training_params
        # )

        if informed_generation:
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
            elif "RedPajama" in tokeniser.name_or_path:
                logger.debug("'RedPajama' tokeniser chosen, adding 178 to vocab size for logits processor.")
                vocab_size += 178
            elif model_params["llama"] or "falcon" in tokeniser.name_or_path:
                vocab_size = len(tokeniser)

            logits_processor.append(
                InformedNERDecoderLogitsProcessor(
                    entity_type_tokens=entity_type_tokens,
                    vocab_size=vocab_size,
                    tokeniser=tokeniser,
                    combine_token=combine_token,
                    entity_separator_token=entity_separator_token,
                    type_content_separator_token=type_content_separator_token,
                    batch_size=training_params["data_loading"]["batch_size"],
                    leading_space=not model_params["llama"],
                )
            )
        else:
            logits_processor = None

        if self.resource.cuda:
            accelerator = "gpu"
            gpus = self.resource.device
            if isinstance(gpus, list) and len(gpus) > 1:
                strategy = FSDPStrategy(cpu_offload=True)
            else:
                strategy = "auto"
        else:
            accelerator = "cpu"
            gpus = "auto"
            strategy = "auto"

        trainer = pl.Trainer(
            devices=gpus,
            accelerator=accelerator,
            logger=loggers,
            callbacks=[ProgressBar()],
            plugins=FluidmlCheckpointIO(task=self),
            strategy=strategy,
        )

        # model = GenerativeNERModel(
        #     model_params=model_params,
        #     optimiser_params=training_params["optimiser"],
        #     generation_params=generation_params,
        #     # evaluator_params=self.training_params["metrics"],
        #     tokeniser=tokeniser,
        #     logits_processor=logits_processor,
        #     is_multigpu=True if strategy != "auto" else False,
        #     is_mainprocess=not self.is_subprocess,
        #     entity_set=corpus.entity_set,
        #     pad_token_id=pad_token_id,
        #     # do_logging=self.is_subprocess,
        # )
        init_model_parameter = {
            "model_params": model_params,
            "optimiser_params": training_params["optimiser"],
            "generation_params": generation_params,
            "tokeniser": tokeniser,
            "logits_processor": logits_processor,
            "is_multigpu": True if strategy != "auto" else False,
            "is_mainprocess": not self.is_subprocess,
            "entity_set": corpus.entity_set,
            "pad_token_id": pad_token_id,
            "type_content_separator_token": type_content_separator_token,
            "entity_separator_token": entity_separator_token,
        }

        if strategy == "auto":
            model = GenerativeNERModel(**init_model_parameter)
        elif isinstance(strategy, FSDPStrategy):
            model = GenerativeNERModelFSDP(**init_model_parameter)
        elif isinstance(strategy, DeepSpeedStrategy):
            model = GenerativeNERModelDeepSpeed(**init_model_parameter)
        else:
            raise ValueError()

        model.load_state_dict(best_model["state_dict"])

        metrics = {}
        for split in datasets.keys():

            logger.info(f"Testing on split '{split}'")
            metrics[split] = trainer.test(model=model, dataloaders=dataloaders[split], verbose=True)
            logger.info(f"{split} performance:")
            logger.info(metrics[split])

        self.save(metrics, "metrics", type_="pickle")
