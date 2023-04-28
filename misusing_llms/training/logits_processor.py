from typing import List

import torch
from torch import LongTensor, FloatTensor, BoolTensor
from transformers import LogitsProcessor, PreTrainedTokenizer


class InformedNERDecoderLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokeniser: PreTrainedTokenizer,
        entity_type_tokens: List[str],
        vocab_size: int,
        combine_token: str,
        entity_separator_token: str,
        type_content_separator_token: str,
        mask_value: int = -1000,
    ):
        self.tokeniser = tokeniser
        self.entity_type_tokens = entity_type_tokens
        self.combine_token = combine_token
        self.entity_separator_token = entity_separator_token
        self.type_content_separator_token = type_content_separator_token
        self.vocab_size = vocab_size
        self.mask_value = mask_value

        self.entity_separator_token_ids = self._tokenise_with_leading_space(text=self.entity_separator_token)
        self.combine_token_ids = self._tokenise_with_leading_space(text=self.combine_token)
        self.type_content_separator_token_ids = self._tokenise_with_leading_space(
            text=self.type_content_separator_token
        )
        self.entity_type_token_ids = [self._tokenise_with_leading_space(text=ett) for ett in self.entity_type_tokens]

        self._mask_rule1 = None
        self._mask_rule2 = None

    def _tokenise_with_leading_space(self, text: str) -> List[List[int]]:
        """tokenises the str input, once without a leading space and once with leading space"""
        without_leading_space = self.tokeniser(text=text).input_ids
        with_leading_space = self.tokeniser(text=" " + text).input_ids
        if with_leading_space == without_leading_space:
            return [without_leading_space]
        else:
            return [without_leading_space, with_leading_space]

    @property
    def mask_rule1(self) -> BoolTensor:
        if self._mask_rule1 is None:
            mask = torch.ones(self.vocab_size, dtype=torch.bool)
            for token_ids in self.entity_type_token_ids:
                mask[token_ids[0][0]] = False
                mask[token_ids[1][0]] = False
            self._mask_rule1 = mask
        return self._mask_rule1

    def __call__(self, input_ids: LongTensor, scores: FloatTensor):
        # Enforced rules for NER generation:
        #   1.  After the "combine_token" or the entity separator (";") the first token of an entity type has to be
        #       predicted.
        #   2.  After predicting the last token of an entity type, the type-content separator (":") has to be predicted.
        #   3.  During the entity content prediction phase, i.e. after ":" and before ";" has been predicted, only
        #       token_ids present in the input_ids are allowed for prediction.

        # save the previous token for easier access
        previous_token_ids = input_ids[:, -1]

        # loop through batch
        for i, previous_token_id in enumerate(previous_token_ids.tolist()):
            if previous_token_id in (self.combine_token_id, self.entity_separator_token_id):
                # case 1:
                #   After the "combine_token" or the entity separator (";") the first token of an entity type has to be
                #   predicted.
                pass
        pass
