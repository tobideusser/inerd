from typing import Dict, Optional, List

import pytorch_lightning as pl
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from misusing_llms.training import Optimiser, LearningRateScheduler, Evaluator


class GenerativeNERModel(pl.LightningModule):
    def __init__(
        self,
        model_params: Dict,
        tokeniser: PreTrainedTokenizerFast,
        optimiser_params: Optional[Dict] = None,
        lr_scheduler_params: Optional[Dict] = None,
        evaluator_params: Optional[Dict] = None,
        evaluator: Optional[Evaluator] = None,
    ):
        super().__init__()

        model_name = model_params.pop("model_name")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokeniser = tokeniser

        self.optimiser_params = optimiser_params
        self.lr_scheduler_params = lr_scheduler_params
        if evaluator is not None:
            self.evaluator = evaluator
        elif evaluator_params is not None:
            self.evaluator = Evaluator.from_config(**evaluator_params)

    def _convert_output_to_entities(self, output_tokens: List[List[str]], labels: List[List[int]]):
        # loop over each element in the batch
        for ot, l in zip(output_tokens, labels):
            entity_string = [token for token, label in zip(ot, l) if label != -100]
            a = self.tokeniser.decode(self.tokeniser.encode(entity_string))

    def generate(self, batch) -> Dict:
        # add "informed" greedy decoding? like in kpi bert?
        return self.model.generate(input_ids=batch, num_beams=1, do_sample=False)  # greedy decoding for now

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
        self.evaluator.update(step_output, split="train")
        self.log(
            "train-loss-step",
            step_output["loss"],
            batch_size=self.trainer.train_dataloader.loaders.batch_size,
        )

    def training_epoch_end(self, outputs: Dict) -> None:
        # compute and log metrics
        metrics = self.evaluator.compute(reset=True, split="train")
        self.log_metrics(metrics)

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
