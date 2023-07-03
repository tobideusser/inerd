from typing import Optional

from datasets import load_dataset, DatasetDict
from tqdm import tqdm

from misusing_llms.data_classes import Sentence, NERCorpus
from misusing_llms.parser import BaseParser


class FiNERORDHuggingFaceParser(BaseParser):
    def __init__(
        self,
        entity_separator_token: str,
        type_content_separator_token: str,
        type_mapping: bool = True,
        debug_size: Optional[int] = None,
        dataset_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        if type_mapping:
            type_mapping = {"PER": "Person", "LOC": "Location", "ORG": "Organisation"}
        else:
            type_mapping = None
        super().__init__(
            type_mapping=type_mapping,
            debug_size=debug_size,
            entity_separator_token=entity_separator_token,
            type_content_separator_token=type_content_separator_token,
            cache_dir=cache_dir,
        )
        self.dataset_name = dataset_name if dataset_name else "FiNER-ORD"

    def _parse_finer_ord_split(self, dataset: DatasetDict, split_type: str):

        # first, organise the dataset in a proper manner
        reordered = dict()
        i = 0
        for token in tqdm(
            dataset[split_type],
            desc=f"Transforming {split_type}",
            total=None if self.debug_size else len(dataset[split_type]),
        ):
            if token["gold_token"] is not None:
                if (token["doc_idx"], token["sent_idx"]) not in reordered:
                    i += 1
                    if self.debug_size is not None and i > self.debug_size:
                        break
                    reordered[(token["doc_idx"], token["sent_idx"])] = {
                        "doc_id": token["doc_idx"],
                        "sent_id": token["sent_idx"],
                        "ner_tags": [token["gold_label"]],
                        "tokens": [token["gold_token"]],
                    }
                else:
                    reordered[(token["doc_idx"], token["sent_idx"])]["ner_tags"].append(token["gold_label"])
                    reordered[(token["doc_idx"], token["sent_idx"])]["tokens"].append(token["gold_token"])

        parsed = []
        for (doc_id, sent_id), sentence in tqdm(
            reordered.items(),
            desc=f"Parsing {split_type}",
            total=len(reordered),
        ):
            entities = self._entity_tags_to_entity_dict(entity_tags=sentence["ner_tags"], words=sentence["tokens"])
            parsed.append(
                Sentence(
                    id_=split_type + "-" + str(doc_id) + "-" + str(sent_id),
                    words=sentence["tokens"],
                    entity_tags=sentence["ner_tags"],
                    entity_label=[self.entity_tag_to_label[entity_tag] for entity_tag in sentence["ner_tags"]],
                    entities_anno=entities,
                    entity_separator_token=self.entity_separator_token,
                    type_content_separator_token=self.type_content_separator_token,
                )
            )
        return parsed

    def parse(self) -> NERCorpus:
        dataset = load_dataset("gtfintechlab/finer-ord", cache_dir=self.cache_dir)
        corpus = {"train": [], "validation": [], "test": []}
        # from https://huggingface.co/datasets/gtfintechlab/finer-ord
        self.entity_tag_to_label = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-LOC",
            4: "I-LOC",
            5: "B-ORG",
            6: "I-ORG",
        }
        if self.type_mapping is not None:
            for k, v in self.entity_tag_to_label.items():
                for kk, vv in self.type_mapping.items():
                    if kk in v:
                        self.entity_tag_to_label[k] = self.entity_tag_to_label[k].replace(kk, vv)
        for split_type in ["train", "validation", "test"]:
            corpus[split_type] = self._parse_finer_ord_split(dataset=dataset, split_type=split_type)
        corpus_parsed = NERCorpus(
            train=corpus["train"],
            validation=corpus["validation"],
            test=corpus["test"],
            name=self.dataset_name,
            entity_tag_to_label=self.entity_tag_to_label,
        )
        return corpus_parsed
