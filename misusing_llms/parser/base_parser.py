from abc import ABC, abstractmethod
from importlib import import_module
from inspect import signature
from typing import List, Dict, Optional

from misusing_llms.data_classes import Entity


class BaseParser(ABC):
    PARSER = {
        "conll2003": "misusing_llms.parser.conll2003_huggingface_parser.CoNLL2003HuggingFaceParser",
        "CoNLL-2003": "misusing_llms.parser.conll2003_huggingface_parser.CoNLL2003HuggingFaceParser",
    }

    def __init__(self, type_mapping: Optional[Dict] = None):
        self.type_mapping = type_mapping

        self.entity_tag_to_label = None
        self._begin_tags = None

    @property
    def begin_tags(self) -> List[int]:
        if not self._begin_tags:
            self._begin_tags = []
            for k, v in self.entity_tag_to_label.items():
                if v.split("-")[0] == "B":
                    self._begin_tags.append(k)
        return self._begin_tags

    def _entity_tags_to_entity_dict(self, entity_tags: List, words: List) -> List[Entity]:
        entity_found_flag = False
        entities = []
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
                    entity = Entity(start=i, words=[words[i]], type_=self.entity_tag_to_label[entity_tag].split("-")[1])
            if (i + 1) == len(entity_tags) and entity_found_flag:
                entities.append(entity)

        return entities

    @abstractmethod
    def parse(self):
        raise NotImplementedError

    @classmethod
    def load_parser(cls, type_: str, type_mapping: Optional[Dict] = None, *args, **kwargs) -> "BaseParser":
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
        return class_(type_mapping=type_mapping, *args, **kwargs_filtered)
