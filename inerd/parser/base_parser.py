from abc import ABC, abstractmethod
from importlib import import_module
from inspect import signature
from typing import List, Dict, Optional

from inerd.data_classes import Entity


class BaseParser(ABC):
    PARSER = {
        "conll2003": "inerd.parser.conll2003_huggingface_parser.CoNLL2003HuggingFaceParser",
        "CoNLL-2003": "inerd.parser.conll2003_huggingface_parser.CoNLL2003HuggingFaceParser",
        "CoNLL2003": "inerd.parser.conll2003_huggingface_parser.CoNLL2003HuggingFaceParser",
        "BC5CDR": "inerd.parser.bc5cdr_parser.BC5CDRParser",
        "OntoNotes": "inerd.parser.ontonotes_huggingface_parser.OntoNotesHuggingFaceParser",
        "NCBI-disease": "inerd.parser.ncbi_disease_huggingface_parser.NCBIDiseaseHuggingFaceParser",
        "WNUT-17": "inerd.parser.wnut17_huggingface_parser.WNUT17HuggingFaceParser",
        "JNLPBA": "inerd.parser.jnlpba_huggingface_parser.JNLPBAHuggingFaceParser",
        "Few-NERD": "inerd.parser.few_nerd_huggingface_parser.FewNERDHuggingFaceParser",
        "CoNLL++": "inerd.parser.conllpp_huggingface_parser.CoNLLPlusPlusHuggingFaceParser",
        "FiNER-ORD": "inerd.parser.finer_ord_huggingface_parser.FiNERORDHuggingFaceParser",
        # "Species-800": "inerd.parser.species_800_huggingface_parser.Species800HuggingFaceParser",
    }

    def __init__(
        self,
        entity_separator_token: str,
        type_content_separator_token: str,
        type_mapping: Optional[Dict] = None,
        debug_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.type_mapping = type_mapping
        self.debug_size = debug_size
        self.entity_separator_token = entity_separator_token
        self.type_content_separator_token = type_content_separator_token
        self.cache_dir = cache_dir

        self.entity_tag_to_label = None
        self._begin_tags = None
        self._outside_tag_id = None

    @property
    def begin_tags(self) -> List[int]:
        if not self._begin_tags:
            self._begin_tags = []
            for k, v in self.entity_tag_to_label.items():
                if v.split("-")[0] == "B":
                    self._begin_tags.append(k)
        return self._begin_tags

    @property
    def outside_tag_id(self) -> int:
        if self._outside_tag_id is None:
            for k, v in self.entity_tag_to_label.items():
                if v == "O" or v == "o":
                    self._outside_tag_id = k
                    break
        return self._outside_tag_id

    def _entity_tags_to_entity_dict(self, entity_tags: List, words: List, tagging_type="iob") -> List[Entity]:
        entity_found_flag = False
        entities = []

        if tagging_type == "iob":
            for i, entity_tag in enumerate(entity_tags):
                if entity_tag in self.begin_tags and not entity_found_flag:
                    entity = Entity(start=i, words=[words[i]], type_=self.entity_tag_to_label[entity_tag].split("-")[1])
                    entity_found_flag = True
                elif entity_found_flag:
                    if entity_tag not in self.begin_tags:
                        try:
                            type_ = self.entity_tag_to_label[entity_tag].split("-")[1]
                            if type_ == entity.type_:
                                entity.words.append(words[i])
                            else:
                                raise Exception
                        except IndexError:
                            entity.end = i
                            entities.append(entity)
                            entity_found_flag = False
                    else:
                        entity.end = i
                        entities.append(entity)
                        entity = Entity(
                            start=i, words=[words[i]], type_=self.entity_tag_to_label[entity_tag].split("-")[1]
                        )
                if (i + 1) == len(entity_tags) and entity_found_flag:
                    entities.append(entity)

        elif tagging_type == "simple":
            for i, entity_tag in enumerate(entity_tags):
                if entity_tag != self.outside_tag_id:
                    if not entity_found_flag:
                        entity = Entity(start=i, words=[words[i]], type_=self.entity_tag_to_label[entity_tag])
                        entity_found_flag = True
                    else:
                        if self.entity_tag_to_label[entity_tag] == entity.type_:
                            entity.words.append(words[i])
                        else:
                            entity.end = i
                            entities.append(entity)
                            entity = Entity(start=i, words=[words[i]], type_=self.entity_tag_to_label[entity_tag])
                elif entity_found_flag:
                    entity.end = i
                    entities.append(entity)
                    entity_found_flag = False

                if (i + 1) == len(entity_tags) and entity_found_flag:
                    entities.append(entity)

        else:
            raise NotImplementedError(f"tagging_type '{tagging_type}' not implemented.")

        return entities

    @abstractmethod
    def parse(self):
        raise NotImplementedError

    @classmethod
    def load_parser(cls, type_: str, *args, **kwargs) -> "BaseParser":
        """function to load different parsers"""
        try:
            callable_path = cls.PARSER[type_]
            parts = callable_path.split(".")
            module_name = ".".join(parts[:-1])
            class_name = parts[-1]
        except KeyError:
            raise KeyError(f'Dataset parser "{type_}" is not implemented.')

        module = import_module(module_name)
        class_ = getattr(module, class_name)
        kwargs_filtered = {k: v for k, v in kwargs.items() if k in signature(class_).parameters}
        return class_(*args, **kwargs_filtered)
