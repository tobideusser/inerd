from typing import Optional

from datasets import load_dataset, DatasetDict
from tqdm import tqdm

from misusing_llms.data_classes import Sentence, NERCorpus
from misusing_llms.parser import BaseParser


class OntoNotesHuggingFaceParser(BaseParser):
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
            type_mapping = {
                "PERSON": "Person",
                "NORP": "Group",
                "FAC": "Facility",
                "ORG": "Organization",
                "GPE": "Geopolitical",
                "LOC": "Location",
                "PRODUCT": "Product",
                "DATE": "Date",
                "TIME": "Time",
                "PERCENT": "Percent",
                "MONEY": "Money",
                "QUANTITY": "Quantity",
                "ORDINAL": "Ordinal",
                "CARDINAL": "Cardinal",
                "EVENT": "Event",
                "WORK_OF_ART": "Art",
                "LAW": "Law",
                "LANGUAGE": "Language",
            }
        else:
            type_mapping = None
        super().__init__(
            type_mapping=type_mapping,
            debug_size=debug_size,
            entity_separator_token=entity_separator_token,
            type_content_separator_token=type_content_separator_token,
            cache_dir=cache_dir,
        )
        self.dataset_name = dataset_name if dataset_name else "OntoNotes"

    def _parse_ontonote_split(self, dataset: DatasetDict, split_type: str):
        parsed = []
        i = 0
        for document in tqdm(
            dataset[split_type],
            desc=f"Parsing {split_type}",
            total=len(dataset[split_type]),
        ):
            for sentence in document["sentences"]:
                entities = self._entity_tags_to_entity_dict(
                    entity_tags=sentence["named_entities"], words=sentence["words"]
                )
                parsed.append(
                    Sentence(
                        id_=document["document_id"] + "-" + str(sentence["part_id"]),
                        words=sentence["words"],
                        entity_tags=sentence["named_entities"],
                        entity_label=[
                            self.entity_tag_to_label[entity_tag] for entity_tag in sentence["named_entities"]
                        ],
                        entities_anno=entities,
                        entity_separator_token=self.entity_separator_token,
                        type_content_separator_token=self.type_content_separator_token,
                    )
                )
                i += 1
                if self.debug_size and i >= self.debug_size:
                    return parsed
        return parsed

    def parse(self) -> NERCorpus:
        dataset = load_dataset("conll2012_ontonotesv5", "english_v12", cache_dir=self.cache_dir)
        corpus = {"train": [], "validation": [], "test": []}
        entity_labels = dataset["train"].features["sentences"][0]["named_entities"].feature.names
        self.entity_tag_to_label = {i: entity_label for i, entity_label in enumerate(entity_labels)}
        if self.type_mapping is not None:
            for k, v in self.entity_tag_to_label.items():
                for kk, vv in self.type_mapping.items():
                    if kk in v:
                        self.entity_tag_to_label[k] = self.entity_tag_to_label[k].replace(kk, vv)
        for split_type in ["train", "validation", "test"]:
            corpus[split_type] = self._parse_ontonote_split(dataset=dataset, split_type=split_type)

        corpus_parsed = NERCorpus(
            train=corpus["train"],
            validation=corpus["validation"],
            test=corpus["test"],
            name=self.dataset_name,
            entity_tag_to_label=self.entity_tag_to_label,
        )
        return corpus_parsed
