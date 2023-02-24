from abc import ABC, abstractmethod
from importlib import import_module
from inspect import signature


class BaseParser(ABC):
    PARSER = {
        "conll2003": "wimp.parser.conll2003_huggingface_parser.CoNLL2003HuggingFaceParser",
        "CoNLL-2003": "wimp.parser.conll2003_huggingface_parser.CoNLL2003HuggingFaceParser",
    }

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
