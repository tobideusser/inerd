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

        # divide all entity tokens in either begin and mid or end, this is to make the calculations for rule 2 & 3
        # easier
        self.entity_type_token_ids_begin_and_mid = []
        self.entity_type_token_ids_end = []
        for entity_type_token_ids in self.entity_type_token_ids:
            for entity_type_token_id in entity_type_token_ids:
                if len(entity_type_token_id) == 1:
                    self.entity_type_token_ids_end.append(entity_type_token_id)
                else:
                    self.entity_type_token_ids_end.append([entity_type_token_id[-1]])
                    self.entity_type_token_ids_begin_and_mid.append(entity_type_token_id[:-1])

        self.mask_rule1 = torch.ones(self.vocab_size, dtype=torch.bool)
        for token_ids in self.entity_type_token_ids:
            self.mask_rule1[token_ids[0][0]] = False
            self.mask_rule1[token_ids[1][0]] = False

        self.mask_rule3 = torch.ones(self.vocab_size, dtype=torch.bool)
        self.mask_rule3[self.entity_separator_token_ids[0][0]] = False
        self.mask_rule3[self.entity_separator_token_ids[1][0]] = False

    def _tokenise_with_leading_space(self, text: str) -> List[List[int]]:
        """tokenises the str input, once without a leading space and once with leading space"""
        without_leading_space = self.tokeniser(text=text).input_ids
        with_leading_space = self.tokeniser(text=" " + text).input_ids
        if with_leading_space == without_leading_space:
            return [without_leading_space]
        else:
            return [without_leading_space, with_leading_space]

    def __call__(self, input_ids: LongTensor, scores: FloatTensor):
        # Enforced rules for NER generation:
        #   1.  After the "combine_token" or the entity separator (";") the first token of an entity type has to be
        #       predicted.
        #   2.  After predicting the first token of any entity type token, the entity type must be completed before any
        #       other tokens are predicted.
        #   3.  After predicting the last token of an entity type, the type-content separator (":") has to be predicted.
        #   4.  During the entity content prediction phase, i.e. after ":" and before ";" has been predicted, only
        #       token_ids present in the input_ids and the entity separator (";") are allowed for prediction. This rule
        #       is divided into to "sub-rules":
        #           4a. After the type-content separator (":") any token from the input may be predicted.
        #           4b. After a token from the input has been predicted, the only allowed tokens for prediction are
        #               either the entity separator (";") or the token following the previous token in the input.

        # save the previous token for easier access
        previous_token_ids = input_ids[:, -1]

        # save the device the scores are on (again for easier access)
        device = scores.device

        # loop through batch
        for i, previous_token_id in enumerate(previous_token_ids.tolist()):
            if [previous_token_id] in self.combine_token_ids + self.entity_separator_token_ids:
                # case 1:
                #   After the "combine_token" or the entity separator (";") the first token of an entity type has to be
                #   predicted.
                scores[i].masked_fill_(mask=self.mask_rule1.to(device), value=self.mask_value)
            elif [previous_token_id] in self.entity_type_token_ids_begin_and_mid:
                # case 2:
                #   After predicting the first token of any entity type token, the entity type must be completed before
                #   any other tokens are predicted.
                pass
            elif [previous_token_id] in self.entity_type_token_ids_end:
                # case 3:
                #   After predicting the last token of an entity type, the type-content separator (":") has to be
                #   predicted.
                scores[i].masked_fill_(mask=self.mask_rule3.to(device), value=self.mask_value)
            elif [previous_token_id] in self.type_content_separator_token_ids:
                # case 4a:
                #   After the type-content separator (":") any token from the input may be predicted.
                pass
        pass
