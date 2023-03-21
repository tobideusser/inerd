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

    #     # used for to store values for informed generation, thus prefixed with ig
    #     self._ig_generation_config = None
    #
    # @property
    # def ig_generation_config(self) -> GenerationConfig:
    #     if self._ig_generation_config is None:
    #         self._ig_generation_config = GenerationConfig.from_model_config(self.model.config)
    #     return self._ig_generation_config
    #
    # def informed_generate(self, **kwargs) -> torch.LongTensor:
    #     """mostly copied from huggingface's implementation of generate"""
    #
    #     # ------------------------------------------------------------------
    #     # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    #     generation_config = copy.deepcopy(self.ig_generation_config)
    #     model_kwargs = generation_config.update(**kwargs)
    #     self._validate_model_kwargs(model_kwargs.copy())
    #
    #     # ------------------------------------------------------------------
    #     # 2. Set generation parameters
    #     logits_processor = LogitsProcessorList()
    #     stopping_criteria = StoppingCriteriaList()
    #
    #     # ------------------------------------------------------------------
    #     # 3. Define model inputs
    #     # inputs_tensor has to be defined
    #     # model_input_name is defined if model-specific keyword input is passed
    #     # otherwise model_input_name is None
    #     # all model-specific keyword inputs are removed from `model_kwargs`
    #     inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
    #         bos_token_id=generation_config.bos_token_id, model_kwargs=model_kwargs
    #     )
    #     batch_size = inputs_tensor.shape[0]
    #
    #     # ------------------------------------------------------------------
    #     # 4. Define other model kwargs
    #     model_kwargs["output_attentions"] = generation_config.output_attentions
    #     model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    #     model_kwargs["use_cache"] = generation_config.use_cache
    #
    #     accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    #     requires_attention_mask = "encoder_outputs" not in model_kwargs
    #
    #     if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
    #         model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
    #             inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
    #         )
    #
    #     # decoder-only models should use left-padding for generation
    #     if not self.config.is_encoder_decoder:
    #         if (
    #             generation_config.pad_token_id is not None
    #             and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
    #         ):
    #             logger.warning(
    #                 "A decoder-only architecture is being used, but right-padding was detected! For correct "
    #                 "generation results, please set `padding_side='left'` when initializing the tokenizer."
    #             )
    #
    #     if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
    #         # if model is encoder decoder encoder_outputs are created
    #         # and added to `model_kwargs`
    #         model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
    #             inputs_tensor, model_kwargs, model_input_name
    #         )
    #
    #     # ------------------------------------------------------------------
    #     # 5. Prepare `input_ids` which will be used for auto-regressive generation
    #     if self.config.is_encoder_decoder:
    #         input_ids = self._prepare_decoder_input_ids_for_generation(
    #             batch_size,
    #             decoder_start_token_id=generation_config.decoder_start_token_id,
    #             bos_token_id=generation_config.bos_token_id,
    #             model_kwargs=model_kwargs,
    #             device=inputs_tensor.device,
    #         )
    #     else:
    #         # if decoder-only then inputs_tensor has to be `input_ids`
    #         input_ids = inputs_tensor
    #
    #     # ------------------------------------------------------------------
    #     # 6. Prepare `max_length` depending on other stopping criteria.
    #     input_ids_seq_length = input_ids.shape[-1]
    #     has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    #     if has_default_max_length and generation_config.max_new_tokens is None:
    #         logger.warning(
    #             "Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to"
    #             f" {generation_config.max_length} (`generation_config.max_length`). Controlling `max_length` via the"
    #             " config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we"
    #             " recommend using `max_new_tokens` to control the maximum length of the generation.",
    #         )
    #     elif has_default_max_length and generation_config.max_new_tokens is not None:
    #         generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
    #     elif not has_default_max_length and generation_config.max_new_tokens is not None:
    #         raise ValueError(
    #             "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
    #             " limit to the generated output length. Remove one of those arguments. Please refer to the"
    #             " documentation for more information. "
    #             "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
    #         )
    #
    #     if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
    #         raise ValueError(
    #             f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
    #             f" the maximum length ({generation_config.max_length})"
    #         )
    #     if input_ids_seq_length >= generation_config.max_length:
    #         input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    #         logger.warning(
    #             f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
    #             f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
    #             " increasing `max_new_tokens`."
    #         )
    #
    #     # ------------------------------------------------------------------
    #     # 7. determine generation mode
    #     is_greedy_gen_mode = True
    #
    #     # ------------------------------------------------------------------
    #     # 8. prepare distribution pre_processing samplers
    #     logits_processor = self._get_logits_processor(
    #         generation_config=generation_config,
    #         input_ids_seq_length=input_ids_seq_length,
    #         encoder_input_ids=inputs_tensor,
    #         prefix_allowed_tokens_fn=None,
    #         logits_processor=logits_processor,
    #     )

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
