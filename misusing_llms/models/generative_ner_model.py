from typing import Dict, Optional

import pytorch_lightning as pl
from transformers import AutoModelForCausalLM

from misusing_llms.training import Optimiser, LearningRateScheduler, Evaluator


class GenerativeNERModel(pl.LightningModule):
    def __init__(
        self,
        model_params: Dict,
        optimiser_params: Optional[Dict] = None,
        lr_scheduler_params: Optional[Dict] = None,
        evaluator_params: Optional[Dict] = None,
    ):
        super().__init__()

        model_name = model_params.pop("model_name")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.optimiser_params = optimiser_params
        self.lr_scheduler_params = lr_scheduler_params
        if evaluator_params is not None:
            self.evaluator = Evaluator.from_config(**evaluator_params)

    def forward(self, input_ids) -> Dict:
        return self.model.generate(input_ids=input_ids, num_beams=1, do_sample=False)  # greedy decoding for now

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.forward(batch)

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
