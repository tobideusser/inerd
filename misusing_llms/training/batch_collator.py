from typing import Tuple, Dict, Any

import torch

from misusing_llms.data_classes import Sentence


class NERBatchCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Tuple[Sentence, ...]) -> Dict[str, Any]:
        # max_length_tokens = max([sentence.num_tokens for sentence in batch])
        # max_length_entity_string_tokens = max([len(sentence.entity_string_token_ids) for sentence in batch])
        max_length_input_ids = max([sentence.num_input_ids for sentence in batch])
        d = {
            "input_ids": torch.stack(
                [
                    torch.nn.functional.pad(
                        input=torch.tensor(sentence.input_ids),
                        pad=(0, max_length_input_ids - sentence.num_input_ids),
                        value=self.pad_token_id,
                    )
                    for sentence in batch
                ]
            ),
            "labels": torch.stack(
                [
                    torch.nn.functional.pad(
                        input=torch.tensor(sentence.labels),
                        pad=(0, max_length_input_ids - sentence.num_input_ids),
                        value=self.pad_token_id,
                    )
                    for sentence in batch
                ]
            ),
            "input_tokens": [sentence.input_tokens for sentence in batch],
            "entity_string": [sentence.entity_string for sentence in batch],
        }
        return d
