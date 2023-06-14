import os
import xml.etree.ElementTree as Et
from typing import Optional, Dict

from misusing_llms.data_classes import Sentence, NERCorpus, Entity
from misusing_llms.parser import BaseParser


class BC5CDRParser(BaseParser):
    def __init__(
        self,
        path_to_data_folders: str,
        entity_separator_token: str,
        type_content_separator_token: str,
        train_file_name: str = "CDR_TrainingSet.BioC.xml",
        valid_file_name: str = "CDR_DevelopmentSet.BioC.xml",
        test_file_name: str = "CDR_TestSet.BioC.xml",
        debug_size: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ):
        super().__init__(
            type_mapping=None,
            debug_size=debug_size,
            entity_separator_token=entity_separator_token,
            type_content_separator_token=type_content_separator_token,
        )
        self.dataset_name = dataset_name if dataset_name else "BC5CDR"
        self.file_paths = {
            "train": os.path.join(path_to_data_folders, train_file_name),
            "valid": os.path.join(path_to_data_folders, valid_file_name),
            "test": os.path.join(path_to_data_folders, test_file_name),
        }

    def _parse_bc5cdr_tree(self, tree: Et.ElementTree):
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
                    data.append(
                        Sentence(
                            text=text,
                            id_=document_id + "-" + str(ii),
                            entities_anno=entities,
                            entity_separator_token=self.entity_separator_token,
                            type_content_separator_token=self.type_content_separator_token,
                        )
                    )
                    i += 1
                    if self.debug_size and i >= self.debug_size:
                        return data
        return data

    def parse(self) -> NERCorpus:
        corpus = dict()
        for split, file_path in self.file_paths.items():

            tree = Et.parse(file_path)
            corpus[split] = self._parse_bc5cdr_tree(tree=tree)

        corpus_parsed = NERCorpus(
            train=corpus["train"],
            validation=corpus["valid"],
            test=corpus["test"],
            name=self.dataset_name,
        )
        return corpus_parsed
