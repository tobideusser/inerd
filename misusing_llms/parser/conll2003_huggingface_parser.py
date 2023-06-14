from typing import Optional, Dict

from datasets import load_dataset
from tqdm import tqdm

from misusing_llms.data_classes import Sentence, NERCorpus
from misusing_llms.parser import BaseParser


class CoNLL2003HuggingFaceParser(BaseParser):
    def __init__(
        self,
        entity_separator_token: str,
        type_content_separator_token: str,
        type_mapping: bool = True,
        debug_size: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ):
        if type_mapping:
            type_mapping = {"PER": "Person", "LOC": "Location", "ORG": "Organisation", "MISC": "Miscellaneous"}
        else:
            type_mapping = None
        super().__init__(
            type_mapping=type_mapping,
            debug_size=debug_size,
            entity_separator_token=entity_separator_token,
            type_content_separator_token=type_content_separator_token,
        )
        self.dataset_name = dataset_name if dataset_name else "CoNLL2003"

    def parse(self) -> NERCorpus:
        dataset = load_dataset("conll2003")
        corpus = {"train": [], "validation": [], "test": []}
        # source: https://huggingface.co/datasets/conll2003
        self.entity_tag_to_label = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
            7: "B-MISC",
            8: "I-MISC",
        }
        if self.type_mapping is not None:
            for k, v in self.entity_tag_to_label.items():
                for kk, vv in self.type_mapping.items():
                    if kk in v:
                        self.entity_tag_to_label[k] = self.entity_tag_to_label[k].replace(kk, vv)
        for split_type in ["train", "validation", "test"]:
            for i, sentence in tqdm(
                enumerate(dataset[split_type]),
                desc=f"Parsing {split_type}",
                total=self.debug_size if self.debug_size else len(dataset[split_type]),
            ):
                entities = self._entity_tags_to_entity_dict(entity_tags=sentence["ner_tags"], words=sentence["tokens"])
                corpus[split_type].append(
                    Sentence(
                        id_=sentence["id"],
                        words=sentence["tokens"],
                        entity_tags=sentence["ner_tags"],
                        entity_label=[self.entity_tag_to_label[entity_tag] for entity_tag in sentence["ner_tags"]],
                        entities_anno=entities,
                        entity_separator_token=self.entity_separator_token,
                        type_content_separator_token=self.type_content_separator_token,
                    )
                )
                if self.debug_size and i >= self.debug_size - 1:
                    break
        corpus_parsed = NERCorpus(
            train=corpus["train"],
            validation=corpus["validation"],
            test=corpus["test"],
            name=self.dataset_name,
            entity_tag_to_label=self.entity_tag_to_label,
        )
        return corpus_parsed
