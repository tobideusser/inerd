from typing import Tuple, Dict, Any

import torch

from inerd.data_classes import Sentence


class NERBatchCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Tuple[Sentence, ...]) -> Dict[str, Any]:
        max_length_input_ids = max([sentence.num_input_ids for sentence in batch])
        max_length_prompt_ids = max([sentence.prompt_end_in_input_ids for sentence in batch])
        prompt_ids = [sentence.input_ids[: sentence.prompt_end_in_input_ids] for sentence in batch]

        d = {
            "input_ids": torch.stack(
                [
                    torch.nn.functional.pad(
                        input=torch.tensor(sentence.input_ids),
                        pad=(max_length_input_ids - sentence.num_input_ids, 0),
                        value=self.pad_token_id,
                    )
                    for sentence in batch
                ]
            ),
            "labels": torch.stack(
                [
                    torch.nn.functional.pad(
                        input=torch.tensor(sentence.labels),
                        pad=(max_length_input_ids - sentence.num_input_ids, 0),
                        value=-100,
                    )
                    for sentence in batch
                ]
            ),
            "prompt_ids": torch.stack(
                [
                    torch.nn.functional.pad(
                        input=torch.tensor(prompt_ids[i]),
                        pad=(max_length_prompt_ids - sentence.prompt_end_in_input_ids, 0),
                        value=self.pad_token_id,
                    )
                    for i, sentence in enumerate(batch)
                ]
            ),
            "input_tokens": [sentence.input_tokens for sentence in batch],
            "entity_string": [sentence.entity_string for sentence in batch],
            "ground_truth_entities": [[entity.to_dict() for entity in sentence.entities_anno] for sentence in batch],
            "max_length_prompt_ids": max_length_prompt_ids,
            "max_length_entity_string_tokens": max_length_input_ids - max_length_prompt_ids,
        }
        return d
