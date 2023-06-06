import logging
import os
from copy import deepcopy
from typing import Optional, Union, List, Dict

import pytorch_lightning as pl
from fluidml import Task
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import FSDPStrategy
from transformers import AutoTokenizer, LogitsProcessorList, LlamaTokenizer

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

from misusing_llms.utils import is_debug

logger = logging.getLogger(__name__)


class NEREvaluation(Task):
    def __init__(self, split_types: Optional[Union[str, List]] = None):
        super().__init__()
        if split_types is None:
            split_types = "test"
        if isinstance(split_types, list):
            self.split_types = split_types
        else:
            self.split_types = [split_types]
        self.is_subprocess = "LOCAL_RANK" in os.environ

    def _init_torch_datasets(self, corpus: NERCorpus) -> Dict[str, GenerativeNERDataset]:
        datasets = {}
        if "test" in self.split_types:
            datasets["test"] = GenerativeNERDataset(sentences=corpus.test)
        if "validation" in self.split_types:
            datasets["validation"] = GenerativeNERDataset(sentences=corpus.test)
        if "train" in self.split_types:
            datasets["train"] = GenerativeNERDataset(sentences=corpus.test)
        return datasets

    @staticmethod
    def _init_torch_dataloaders(
        datasets: Dict[str, GenerativeNERDataset],
        batch_collator: NERBatchCollator,
        training_params: Dict,
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
                shuffle=False,
                num_workers=num_workers,
                # drop_last=True,
                **training_params["data_loading"],
            )

        return dataloaders

    def run(self, best_model: Dict, corpus_tokenised: NERCorpus):
        if isinstance(corpus_tokenised, Dict):
            logger.info("Converting corpus_tokenised dict to Corpus object.")
            corpus = NERCorpus.from_dict(corpus_tokenised)
        else:
            corpus = corpus_tokenised

        model_params = deepcopy(self.unique_config["NERTraining"]["model_params"])
        training_params = deepcopy(self.unique_config["NERTraining"]["training_params"])
        generation_params = deepcopy(self.unique_config["NERTraining"]["generation_params"])
        informed_generation = generation_params.pop("informed_generation")

        if (
            "bloom" in model_params["model_name"]
            or "RedPajama" in model_params["model_name"]
            or "opt" in model_params["model_name"]
            or "gpt-2" in model_params["model_name"]
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if model_params["llama"]:
            tokeniser = LlamaTokenizer.from_pretrained(model_params["model_name"])
        else:
            tokeniser = AutoTokenizer.from_pretrained(model_params["model_name"])

        if model_params["llama"]:
            tokeniser.add_special_tokens({"pad_token": "<|padding|>"})
            pad_token_id = tokeniser.pad_token_id
        elif "RedPajama" in model_params["model_name"]:
            pad_token_id = 1  # "<|padding|>" in GPT-NEOX
            tokeniser.pad_token_id = 1
        elif "falcon" in model_params["model_name"]:
            tokeniser.add_special_tokens({"pad_token": "<|padding|>"})
            pad_token_id = tokeniser.pad_token_id
        elif tokeniser.pad_token_id is None:
            raise NotImplementedError
        else:
            pad_token_id = tokeniser.pad_token_id

        batch_collator = NERBatchCollator(pad_token_id=pad_token_id)
        datasets = self._init_torch_datasets(corpus=corpus)
        dataloaders = self._init_torch_dataloaders(
            datasets=datasets, batch_collator=batch_collator, training_params=training_params
        )

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
                    entity_separator_token=";",
                    type_content_separator_token=":",
                    batch_size=training_params["data_loading"]["batch_size"],
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
            gpus=gpus,
            # logger=loggers,
            callbacks=[ProgressBar()],
            plugins=FluidmlCheckpointIO(task=self),
            strategy=strategy,
        )

        model = GenerativeNERModel(
            model_params=model_params,
            optimiser_params=training_params["optimiser"],
            generation_params=generation_params,
            # evaluator_params=self.training_params["metrics"],
            tokeniser=tokeniser,
            logits_processor=logits_processor,
            is_multigpu=True if strategy != "auto" else False,
            is_mainprocess=not self.is_subprocess,
            entity_set=corpus.entity_set,
            pad_token_id=pad_token_id,
            # do_logging=self.is_subprocess,
        )

        model.load_state_dict(best_model["state_dict"])

        metrics = {}
        for split in datasets.keys():

            logger.info(f"Testing on split '{split}'")
            metrics[split] = trainer.test(model=model, dataloaders=dataloaders[split], verbose=True)
