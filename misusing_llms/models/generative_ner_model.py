import os

import logging
from typing import Dict, Optional, List, Set

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from peft import get_peft_model, LoraConfig, TaskType
from torch.distributed.fsdp.wrap import wrap
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning.utilities import rank_zero_only


from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList

from misusing_llms.training import Optimiser, LearningRateScheduler, Evaluator
from misusing_llms.utils import entity_string_to_entity_dataclass

logger = logging.getLogger(__name__)


class GenerativeNERModel(pl.LightningModule):
    def __init__(
        self,
        model_params: Dict,
        tokeniser: PreTrainedTokenizerFast,
        generation_params: Dict,
        pad_token_id: Optional[int] = None,
        is_multigpu: bool = True,
        is_mainprocess: bool = True,
        logits_processor: Optional[LogitsProcessorList] = None,
        optimiser_params: Optional[Dict] = None,
        learning_rate_scheduler_inputs: Optional[Dict] = None,
        evaluator_params: Optional[Dict] = None,
        evaluator: Optional[Evaluator] = None,
        entity_set: Optional[Set[str]] = None,
    ):
        super().__init__()
        self.model_name = model_params["model_name"]
        self.entity_set = entity_set
        self.load_in_8bit: bool = model_params["load_in_8bit"]
        self.lora: bool = model_params["lora"]

        if self.load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        if self.lora:
            self.lora_config = model_params["lora_config"]
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **self.lora_config)
            self.model = get_peft_model(model=self.model, peft_config=peft_config)
        else:
            self.lora_config = None

        self.tokeniser = tokeniser
        if pad_token_id:
            self.pad_token_id = pad_token_id
        else:
            self.pad_token_id = self.tokeniser.pad_token_id
        self.logits_processor = logits_processor

        self.generation_params = generation_params
        self.optimiser_params = optimiser_params
        self.learning_rate_scheduler_inputs = learning_rate_scheduler_inputs
        if evaluator is not None:
            self.evaluator = evaluator
        elif evaluator_params is not None:
            self.evaluator = Evaluator.from_config(**evaluator_params)

        self.is_multigpu = is_multigpu
        self.is_mainprocess = is_mainprocess

        # logging stuff
        self.ground_truth_entities: List[List[dict]] = []
        self.entity_strings_predicted: List[str] = []
        self.entity_strings_ground_truth: List[str] = []
        self.best_valid_ner_micro_f1 = 0
        self.best_epoch = 0

    def configure_sharded_model(self) -> None:
        if self.model.base_model_prefix == "gpt_neox":  # redpajama model
            self.model.gpt_neox.embed_in = wrap(self.model.gpt_neox.embed_in)
            for i, layer in enumerate(self.model.gpt_neox.layers):
                self.model.gpt_neox.layers[i] = wrap(layer)
            self.model.gpt_neox.final_layer_norm = wrap(self.model.gpt_neox.final_layer_norm)
            self.model.embed_out = wrap(self.model.embed_out)
        else:
            raise NotImplementedError(
                f"manual wrapping for model_name: {self.model_name} and base_model_prefix: "
                f"{self.model.base_model_prefix} not implemented."
            )

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        predictions = self.model.generate(
            input_ids=batch["prompt_ids"],
            attention_mask=(batch["prompt_ids"] != self.pad_token_id).type(torch.LongTensor).to(self.device),
            logits_processor=self.logits_processor,
            synced_gpus=self.is_multigpu,
            pad_token_id=self.tokeniser.eos_token_id,
            **self.generation_params,
        )
        batch["predictions"] = predictions
        batch["output_tokens"] = [self.tokeniser.convert_ids_to_tokens(p) for p in torch.unbind(predictions, dim=0)]
        entity_string_token_ids_predicted = predictions[:, batch["max_length_prompt_ids"] :]
        batch["entity_string_predicted"] = self.tokeniser.batch_decode(entity_string_token_ids_predicted)
        batch = self._detach_tensors_in_dict(batch)
        return batch

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        # update stored results to evalaluate them at the end of the epoch
        self.ground_truth_entities.extend(outputs["ground_truth_entities"])
        self.entity_strings_predicted.extend(outputs["entity_string_predicted"])
        self.entity_strings_ground_truth.extend(outputs["entity_string"])

    def on_validation_epoch_end(self) -> None:

        # compute and log metrics
        # metrics = self.evaluator.compute(reset=True)
        self.log_metrics(split="valid")

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.forward(batch)

    def forward(self, batch) -> Dict:
        # return self.model(input_ids=batch["input_ids"], labels=batch.get("labels", None))
        labels = batch.get("labels", None)
        model_output = self.model(input_ids=batch["input_ids"], labels=labels)

        loss = float(model_output.loss)
        self.log(
            "train-loss-step",
            loss,
            batch_size=self.trainer.train_dataloader.batch_size,
            # rank_zero_only=True,
            sync_dist=self.is_multigpu,
        )
        batch["loss"] = model_output.loss
        return batch

    def log_metrics(self, split: str):
        metrics = self.compute_metrics()
        ner_micro_f1 = metrics["ner_micro_f1"]
        logger.info(f"Saved results: {len(self.entity_strings_predicted)} | micro f1: {metrics['ner_micro_f1']}")
        if ner_micro_f1 > self.best_valid_ner_micro_f1:
            self.best_epoch = self.current_epoch
            self.best_valid_ner_micro_f1 = ner_micro_f1
            self.log(
                name="best-" + split + "-ner_micro_f1",
                value=ner_micro_f1,
                sync_dist=self.is_multigpu,
            )
            self.log(
                name="best-epoch",
                value=self.best_epoch,
                sync_dist=self.is_multigpu,
            )
            self._log_predictions()
        for k, v in metrics.items():
            if isinstance(v, dict):
                # self._log_summary_dict(name=split + "-" + k, summary_dict=v)
                pass
            else:
                self.log(
                    name=split + "-" + k,
                    value=v,
                    sync_dist=self.is_multigpu,
                )
        self._reset_predictions()

    @rank_zero_only
    def _log_predictions(self):
        table = pd.DataFrame(
            {
                "ground_truth": self.entity_strings_ground_truth,
                "predicted": self.entity_strings_predicted,
            }
        )
        for train_logger in self.loggers:
            # if isinstance(train_logger, pl.loggers.tensorboard.TensorBoardLogger):
            #     train_logger.experiment.add_text(name, table.to_string(), global_step=self.current_epoch)
            if isinstance(train_logger, pl.loggers.wandb.WandbLogger):
                train_logger.log_text(key="val_predictions", dataframe=table, step=self.global_step)
            else:
                logger.warning(f"pl.Trainer.logger of type {type(train_logger)} can not store text.")

    # def _log_summary_dict(self, name: str, summary_dict: Dict):
    #     # use pandas to format as a human-readable table
    #     table = pd.DataFrame.from_dict(summary_dict, orient="index")
    #     table = table.reset_index(names="metric")
    #     for train_logger in self.loggers:
    #         if isinstance(train_logger, pl.loggers.tensorboard.TensorBoardLogger):
    #             train_logger.experiment.add_text(name, table.to_string(), global_step=self.current_epoch)
    #         elif isinstance(train_logger, pl.loggers.wandb.WandbLogger):
    #             train_logger.log_table(key=name, dataframe=table, step=self.global_step)
    #         else:
    #             logger.warning(f"pl.Trainer.logger of type {type(train_logger)} can not store text.")

    @staticmethod
    def _detach_tensors_in_dict(d: Dict) -> Dict:
        for k, v in d.items():
            if isinstance(v, torch.Tensor) and k != "loss":
                d[k] = v.detach().cpu()
        return d

    def compute_metrics(self):
        assert len(self.ground_truth_entities) == len(self.entity_strings_predicted)

        statistics = {ent: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for ent in self.entity_set}
        clf_report = {}

        predicted_entities = [entity_string_to_entity_dataclass(es) for es in self.entity_strings_predicted]

        # Count TP, FP and FN per type
        for prediction, ground_truth in zip(predicted_entities, self.ground_truth_entities):
            for entity_type in self.entity_set:
                pred_ents = {ent.words for ent in prediction if ent.type_ == entity_type}
                gt_ents = {" ".join(ent["words"]) for ent in ground_truth if ent["type_"] == entity_type}
                statistics[entity_type]["support"] += len(gt_ents)
                statistics[entity_type]["tp"] += len(pred_ents & gt_ents)
                statistics[entity_type]["fp"] += len(pred_ents - gt_ents)
                statistics[entity_type]["fn"] += len(gt_ents - pred_ents)

        # Compute per entity Precision / Recall / F1 / Support
        for entity_type in statistics.keys():
            if statistics[entity_type]["tp"]:
                precision = (
                    100
                    * statistics[entity_type]["tp"]
                    / (statistics[entity_type]["fp"] + statistics[entity_type]["tp"])
                )
                recall = (
                    100
                    * statistics[entity_type]["tp"]
                    / (statistics[entity_type]["fn"] + statistics[entity_type]["tp"])
                )
            else:
                precision, recall = 0.0, 0.0

            if not precision + recall == 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            support = statistics[entity_type]["support"]
            clf_report[entity_type] = {"Precision": precision, "Recall": recall, "F1": f1, "Support": support}

        # Sort clf report descending
        clf_report = dict(sorted(clf_report.items(), key=lambda item: item[1]["Support"], reverse=True))

        # Compute micro F1 Scores
        tp_all = sum([statistics[entity_type]["tp"] for entity_type in self.entity_set])
        fp_all = sum([statistics[entity_type]["fp"] for entity_type in self.entity_set])
        fn_all = sum([statistics[entity_type]["fn"] for entity_type in self.entity_set])
        support_all = sum([statistics[entity_type]["support"] for entity_type in self.entity_set])

        if tp_all:
            micro_precision = 100 * tp_all / (tp_all + fp_all)
            micro_recall = 100 * tp_all / (tp_all + fn_all)
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

        else:
            micro_precision, micro_recall, micro_f1 = 0.0, 0.0, 0.0

        clf_report["micro avg"] = {
            "Precision": micro_precision,
            "Recall": micro_recall,
            "F1": micro_f1,
            "Support": support_all,
        }

        # Compute Macro F1 Scores
        macro_precision = np.mean(
            [
                clf_report[entity_type]["Precision"]
                for entity_type in self.entity_set
                if clf_report[entity_type]["Support"] > 0
            ]
        )
        macro_recall = np.mean(
            [
                clf_report[entity_type]["Recall"]
                for entity_type in self.entity_set
                if clf_report[entity_type]["Support"] > 0
            ]
        )
        macro_f1 = np.mean(
            [clf_report[entity_type]["F1"] for entity_type in self.entity_set if clf_report[entity_type]["Support"] > 0]
        )

        clf_report["macro avg"] = {
            "Precision": macro_precision,
            "Recall": macro_recall,
            "F1": macro_f1,
            "Support": support_all,
        }
        return {"ner_clf_report": clf_report, "ner_micro_f1": micro_f1, "ner_macro_f1": macro_f1}

    def _reset_predictions(self):
        self.entity_strings_predicted = []
        self.ground_truth_entities = []
        self.entity_strings_ground_truth = []

    def configure_optimizers(self):
        optimiser = Optimiser.from_config(
            params=self.trainer.model.parameters(), multigpu=self.is_multigpu, **self.optimiser_params
        )

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
