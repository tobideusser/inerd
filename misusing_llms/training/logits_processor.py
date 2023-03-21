from typing import List

from torch import LongTensor, FloatTensor
from transformers import LogitsProcessor


class InformedNERDecoderLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        entity_type_token_ids: List[List[int]],
        t,
        combine_token_id: int,
        entity_separator_token_id: int,
        type_content_separator_token_id: int,
    ):
        self.entity_type_token_ids = entity_type_token_ids
        self.combine_token_id = combine_token_id
        self.entity_separator_token_id = entity_separator_token_id
        self.type_content_separator_token_id = type_content_separator_token_id
        self.t = t  # remove later, here for debugging reasons

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
        for previous_token_id in previous_token_ids.tolist():
            if previous_token_id in (self.combine_token_id, self.entity_separator_token_id):
                # case 1:
                #   After the "combine_token" or the entity separator (";") the first token of an entity type has to be
                #   predicted.
                pass
        pass
