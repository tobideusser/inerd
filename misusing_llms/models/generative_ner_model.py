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
        logits_processor: Optional[LogitsProcessorList] = None,
        optimiser_params: Optional[Dict] = None,
        lr_scheduler_params: Optional[Dict] = None,
        evaluator_params: Optional[Dict] = None,
        evaluator: Optional[Evaluator] = None,
    ):
        super().__init__()
        model_name = model_params["model_name"]
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokeniser = tokeniser
        self.logits_processor = logits_processor

        self.optimiser_params = optimiser_params
        self.lr_scheduler_params = lr_scheduler_params
        if evaluator is not None:
            self.evaluator = evaluator
        elif evaluator_params is not None:
            self.evaluator = Evaluator.from_config(**evaluator_params)

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

    def validation_step_end(self, step_output: Dict) -> None:
        # update metrics
        self.evaluator.update(self._detach_tensors_in_dict(step_output))

    def validation_epoch_end(self, outputs: Dict) -> None:
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

        batch["logits"] = logits
        batch["loss"] = model_output.loss
        batch["predictions"] = predictions
        batch["output_tokens"] = [self.tokeniser.convert_ids_to_tokens(p) for p in torch.unbind(predictions, dim=0)]

        entity_string_token_ids_predicted = [
            torch.masked_select(p, labels != -100) for p in torch.unbind(predictions, dim=0)
        ]

        batch["entity_string_predicted"] = self.tokeniser.batch_decode(entity_string_token_ids_predicted)

        return batch

    def training_step_end(self, step_output: Dict) -> None:
        # update metrics
        self.evaluator.update(self._detach_tensors_in_dict(step_output))
        loss = float(step_output["loss"])
        self.log(
            "train-loss-step",
            loss,
            batch_size=self.trainer.train_dataloader.loaders.batch_size,
        )

    def training_epoch_end(self, outputs: Dict) -> None:
        # compute and log metrics
        metrics = self.evaluator.compute(reset=True)
        self.log_metrics(metrics, split="train")

    def log_metrics(self, metrics: Dict, split: str):
        for k, v in metrics.items():
            if isinstance(v, dict):
                self._log_summary_dict(name=split + "-" + k, summary_dict=v)
            else:
                self.log(name=split + "-" + k, value=v)

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
                logger.error(f"pl.Trainer.logger of type {type(train_logger)} can not store text.")

    @staticmethod
    def _detach_tensors_in_dict(d: Dict) -> Dict:
        for k, v in d.items():
            if isinstance(v, torch.Tensor) and k != "loss":
                d[k] = v.detach().cpu()
        return d

    def configure_optimizers(self):
        optimiser = Optimiser.from_config(params=self.parameters(), **self.optimiser_params)
        self.trainer.reset_train_dataloader(self)

        if self.lr_scheduler_params is not None:
            total_devices = self.trainer.num_devices * self.trainer.num_nodes
            train_batches = len(self.trainer.train_dataloader) // total_devices
            train_steps = (self.trainer.max_epochs * train_batches) // self.trainer.accumulate_grad_batches
            lr_warmup = self.lr_scheduler_params.pop("lr_warmup", 0.0)
            interval = self.lr_scheduler_params.pop("interval", "epoch")
            lr_scheduler = LearningRateScheduler.from_config(
                optimiser=optimiser,
                num_warmup_steps=lr_warmup * train_steps,
                num_training_steps=train_steps,
                **self.lr_scheduler_params,
            )

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
