from typing import List

from torch import LongTensor, FloatTensor
from transformers import LogitsProcessor


class InformedNERDecoderLogitsProcessor(LogitsProcessor):
    def __init__(self, entity_type_token_ids: List[List[int]], t):
        self.entity_type_token_ids = entity_type_token_ids
        self.t = t  # remove later

    def __call__(self, input_ids: LongTensor, scores: FloatTensor):
        # Enforced rules for NER generation:
        #   1.  After the "combine_token" or the entity separator (";") the first token of an entity type has to be
        #       predicted.
        #   2.  After predicting the last token of an entity type, the type-content separator (":") has to be predicted.
        #   3.  During the entity content prediction phase, i.e. after ":" and before ";" has been predicted, only
        #       token_ids present in the input_ids are allowed for prediction.
        pass
