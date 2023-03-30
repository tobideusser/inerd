# import copy
# import inspect
import logging
from typing import Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList

from misusing_llms.training import Optimiser, LearningRateScheduler, Evaluator

logger = logging.getLogger(__name__)


class GenerativeNERModel(pl.LightningModule):
    def __init__(
        self,
        model_params: Dict,
        tokeniser: PreTrainedTokenizerFast,
        is_multigpu: bool = True,
        logits_processor: Optional[LogitsProcessorList] = None,
        optimiser_params: Optional[Dict] = None,
        learning_rate_scheduler_inputs: Optional[Dict] = None,
        evaluator_params: Optional[Dict] = None,
        evaluator: Optional[Evaluator] = None,
    ):
        super().__init__()
        model_name = model_params["model_name"]

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokeniser = tokeniser
        self.logits_processor = logits_processor

        self.optimiser_params = optimiser_params
        self.learning_rate_scheduler_inputs = learning_rate_scheduler_inputs
        if evaluator is not None:
            self.evaluator = evaluator
        elif evaluator_params is not None:
            self.evaluator = Evaluator.from_config(**evaluator_params)

        self.is_multigpu = is_multigpu

    def generate(self, batch) -> Dict:
        predictions = self.model.generate(
            input_ids=batch["prompt_ids"],
            num_beams=1,
            do_sample=False,
            max_length=batch["max_length_prompt_ids"] + 100,
            logits_processor=self.logits_processor,
        )
        batch["predictions"] = predictions
        batch["output_tokens"] = [self.tokeniser.convert_ids_to_tokens(p) for p in torch.unbind(predictions, dim=0)]
        entity_string_token_ids_predicted = predictions[:, batch["max_length_prompt_ids"] :]
        batch["entity_string_predicted"] = self.tokeniser.batch_decode(entity_string_token_ids_predicted)
        return batch

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        # add "informed" greedy decoding? like in kpi bert?
        return self.generate(batch)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        # update metrics
        self.evaluator.update(self._detach_tensors_in_dict(outputs))

    def on_validation_epoch_end(self) -> None:
        # compute and log metrics
        metrics = self.evaluator.compute(reset=True)
        self.log_metrics(metrics, split="valid")

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.forward(batch)

    def forward(self, batch) -> Dict:
        labels = batch.get("labels", None)
        model_output = self.model(input_ids=batch["input_ids"], labels=labels)

        logits = model_output.logits
        predictions = torch.argmax(logits, dim=-1)

        # batch["logits"] = logits  # logits not needed? just eats all the RAM?
        batch["loss"] = model_output.loss
        batch["predictions"] = predictions
        batch["output_tokens"] = [self.tokeniser.convert_ids_to_tokens(p) for p in torch.unbind(predictions, dim=0)]

        entity_string_token_ids_predicted = [
            torch.masked_select(p, labels != -100) for p in torch.unbind(predictions, dim=0)
        ]

        batch["entity_string_predicted"] = self.tokeniser.batch_decode(entity_string_token_ids_predicted)

        return batch

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # update metrics
        self.evaluator.update(self._detach_tensors_in_dict(outputs))
        loss = float(outputs["loss"])
        self.log(
            "train-loss-step",
            loss,
            batch_size=self.trainer.train_dataloader.batch_size,
            rank_zero_only=self.is_multigpu,
        )

    def on_train_epoch_end(self) -> None:
        # compute and log metrics
        metrics = self.evaluator.compute(reset=True)
        self.log_metrics(metrics, split="train")

    def log_metrics(self, metrics: Dict, split: str):
        for k, v in metrics.items():
            if isinstance(v, dict):
                self._log_summary_dict(name=split + "-" + k, summary_dict=v)
            else:
                self.log(name=split + "-" + k, value=v, rank_zero_only=self.is_multigpu)

    def _log_summary_dict(self, name: str, summary_dict: Dict):
        # use pandas to format as a human-readable table
        table = pd.DataFrame.from_dict(summary_dict, orient="index")
        table = table.reset_index(names="metric")
        for train_logger in self.loggers:
            if isinstance(train_logger, pl.loggers.tensorboard.TensorBoardLogger):
                train_logger.experiment.add_text(name, table.to_string(), global_step=self.current_epoch)
            elif isinstance(train_logger, pl.loggers.wandb.WandbLogger):
                train_logger.log_table(key=name, dataframe=table, step=self.global_step)
            else:
                logger.warning(f"pl.Trainer.logger of type {type(train_logger)} can not store text.")

    @staticmethod
    def _detach_tensors_in_dict(d: Dict) -> Dict:
        for k, v in d.items():
            if isinstance(v, torch.Tensor) and k != "loss":
                d[k] = v.detach().cpu()
        return d

    def configure_optimizers(self):
        optimiser = Optimiser.from_config(params=self.trainer.model.parameters(), **self.optimiser_params)

        if self.learning_rate_scheduler_inputs is not None:
            interval = self.learning_rate_scheduler_inputs.pop("interval", "epoch")
            lr_scheduler = LearningRateScheduler.from_config(optimiser=optimiser, **self.learning_rate_scheduler_inputs)

            scheduler = {
                "scheduler": lr_scheduler,
                "interval": interval,
                "frequency": 1,
                "strict": False,
                "monitor": "loss",
            }

            return [optimiser], [scheduler]
        else:
            return optimiser
