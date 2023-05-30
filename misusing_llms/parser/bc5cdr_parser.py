import os
import xml.etree.ElementTree as ET
from typing import Optional, Dict

from datasets import load_dataset
from tqdm import tqdm

from misusing_llms.data_classes import Sentence, NERCorpus, Entity
from misusing_llms.parser import BaseParser


class BC5CDRParser(BaseParser):
    def __init__(
        self,
        path_to_data_folders: str,
        train_file_name: str = "CDR_TrainingSet.BioC.xml",
        valid_file_name: str = "CDR_DevelopmentSet.BioC.xml",
        test_file_name: str = "CDR_TestSet.BioC.xml",
        type_mapping: Optional[Dict] = None,
        debug_size: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ):
        super().__init__(type_mapping=type_mapping, debug_size=debug_size)
        self.dataset_name = dataset_name if dataset_name else "BC5CDR"
        self.file_paths = {
            "train": os.path.join(path_to_data_folders, train_file_name),
            "valid": os.path.join(path_to_data_folders, valid_file_name),
            "test": os.path.join(path_to_data_folders, test_file_name),
        }

    def _parse_bc5cdr_tree(self, tree: ET.ElementTree):
        root = tree.getroot()
        data = []
        i = 0
        for child in root:
            if child.tag == "document":
                document_id = child.find("id").text
                paraphraps = child.findall("passage")
                for ii, paragraph in enumerate(paraphraps):
                    text = paragraph.find("text").text
                    annotations = paragraph.findall("annotation")
                    paragraph_offset = int(paragraph.find("offset").text)
                    entities = []
                    for annotation in annotations:
                        information = annotation.findall("infon")
                        entity_type = ""
                        for info in information:
                            key = info.attrib.get("key", None)
                            if key == "type":
                                entity_type = info.text
                        if entity_type == "":
                            raise KeyError("No entity type found")
                        location = annotation.find("location")
                        offset = int(location.attrib["offset"]) - paragraph_offset
                        length = int(location.attrib["length"])
                        words = annotation.find("text").text
                        entities.append(
                            Entity(
                                words=words,
                                type_=entity_type,
                                start=offset,
                                end=offset + length,
                            )
                        )
                    data.append(Sentence(text=text, id_=document_id + "-" + str(ii), entities_anno=entities))
                    i += 1
                    if self.debug_size and i >= self.debug_size:
                        return data

    def parse(self) -> NERCorpus:
        corpus = dict()
        for split, file_path in self.file_paths.items():

            tree = ET.parse(file_path)
            corpus[split] = self._parse_bc5cdr_tree(tree=tree)

            pass

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
