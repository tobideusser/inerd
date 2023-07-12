from typing import Optional

from datasets import load_dataset, DatasetDict
from tqdm import tqdm

from inerd.data_classes import Sentence, NERCorpus
from inerd.parser import BaseParser


class JNLPBAHuggingFaceParser(BaseParser):
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
                "DNA": "DNA",
                "protein": "Protein",
                "cell_type": "Cell Type",
                "cell_line": "Cell Line",
                "RNA": "RNA",
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
        self.dataset_name = dataset_name if dataset_name else "JNLPBA"

    def _parse_jnlpba_split(self, dataset: DatasetDict, split_type: str):
        parsed = []
        i = 0
        for sentence in tqdm(
            dataset[split_type],
            desc=f"Parsing {split_type}",
            total=len(dataset[split_type]),
        ):
            entities = self._entity_tags_to_entity_dict(entity_tags=sentence["tags"], words=sentence["tokens"])
            parsed.append(
                Sentence(
                    id_=split_type + "_" + str(i),
                    words=sentence["tokens"],
                    entity_tags=sentence["tags"],
                    entity_label=[self.entity_tag_to_label[entity_tag] for entity_tag in sentence["tags"]],
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
        dataset = load_dataset("tner/bionlp2004", cache_dir=self.cache_dir)
        corpus = {"train": [], "validation": [], "test": []}
        entity_labels = {
            "O": 0,
            "B-DNA": 1,
            "I-DNA": 2,
            "B-protein": 3,
            "I-protein": 4,
            "B-cell_type": 5,
            "I-cell_type": 6,
            "B-cell_line": 7,
            "I-cell_line": 8,
            "B-RNA": 9,
            "I-RNA": 10,
        }
        self.entity_tag_to_label = {v: k for k, v in entity_labels.items()}
        if self.type_mapping is not None:
            for k, v in self.entity_tag_to_label.items():
                for kk, vv in self.type_mapping.items():
                    if kk in v:
                        self.entity_tag_to_label[k] = self.entity_tag_to_label[k].replace(kk, vv)
        for split_type in ["train", "validation", "test"]:
            corpus[split_type] = self._parse_jnlpba_split(dataset=dataset, split_type=split_type)

        corpus_parsed = NERCorpus(
            train=corpus["train"],
            validation=corpus["validation"],
            test=corpus["test"],
            name=self.dataset_name,
            entity_tag_to_label=self.entity_tag_to_label,
        )
        return corpus_parsed
