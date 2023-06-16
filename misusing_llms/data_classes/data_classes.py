from dataclasses import dataclass
from typing import List, Dict, Optional, Iterator, Union, Set

from torch import Tensor


@dataclass
class Index:
    id2value: Dict
    _value2id: Optional[Dict] = None

    @property
    def val2idx(self) -> Dict:
        if not self._value2id:
            self._value2id = dict((value, id_) for id_, value in self.id2value.items())
        return self._value2id


@dataclass
class NamedEntityVocabulary:
    entities: Union[Index, Set[str]]  # for now: IOB tags, might have to be adjusted when using non IOB methods

    @classmethod
    def from_entity_tag_to_label_dict(cls, entity_tag_to_label: Dict):
        return cls(entities=Index(id2value=entity_tag_to_label))

    @classmethod
    def from_corpus(cls, corpus):
        entities_ = set()
        for sentence in corpus.sentences:
            for entity in sentence.entities_anno:
                entities_.add(entity.type_)
        return cls(entities=entities_)


@dataclass
class Entity:
    words: Union[List[str], str]
    type_: str
    start: Optional[int] = None
    end: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)

    def to_dict(self) -> Dict:
        return self.__dict__


@dataclass
class Sentence:
    id_: Union[int, str]

    entity_separator_token: str
    type_content_separator_token: str

    _entity_separator: Optional[str] = None
    _type_content_separator: Optional[str] = None

    entity_tags: Optional[List[int]] = None
    words: Optional[List[str]] = None
    entity_label: Optional[List[str]] = None
    text: Optional[str] = None

    # token_ids: Optional[List[int]] = None
    # tokens: Optional[List[str]] = None

    word2token_alignment_mask: Optional[Union[List[List[bool]], Tensor]] = None
    word2token_start_ids: Optional[List[int]] = None
    word2token_end_ids: Optional[List[int]] = None

    entity_iobes: Optional[List[str]] = None
    entities_anno: Optional[List[Entity]] = None
    # entity_string_tokens: Optional[List[str]] = None
    # entity_string_token_ids: Optional[List[int]] = None

    _content: Optional[str] = None
    _entity_string: Optional[str] = None

    input_ids: Optional[List[int]] = None
    input_tokens: Optional[List[str]] = None
    labels: Optional[List[int]] = None
    _prompt_end_in_input_ids: Optional[int] = None

    def __len__(self):
        return len(self.words)

    @property
    def entity_separator(self) -> str:
        if self._entity_separator is None:
            if len(self.entity_separator_token) == 1 and not self.entity_separator_token.isalnum():
                self._entity_separator = self.entity_separator_token + " "
            else:
                self._entity_separator = " " + self.entity_separator_token + " "
        return self._entity_separator

    @property
    def type_content_separator(self) -> str:
        if self._type_content_separator is None:
            if len(self.type_content_separator_token) == 1 and not self.type_content_separator_token.isalnum():
                self._type_content_separator = self.type_content_separator_token + " "
            else:
                self._type_content_separator = " " + self.type_content_separator_token + " "
        return self._type_content_separator

    @property
    def content(self) -> str:
        if self.text:
            return self.text
        if not self._content:
            self._content = " ".join(self.words)
        return self._content

    @property
    def entity_string(self) -> str:
        if not self._entity_string:
            s = ""
            for entity in self.entities_anno:
                if isinstance(entity.words, str):
                    s += entity.type_ + self.type_content_separator + entity.words + self.entity_separator
                else:
                    s += entity.type_ + self.type_content_separator + " ".join(entity.words) + self.entity_separator

            if len(s) > 0:
                s = s[:-1]

            # if s == "": model should predict EOS token
            self._entity_string = s
        return self._entity_string

    @property
    def prompt_end_in_input_ids(self) -> int:
        if not self._prompt_end_in_input_ids:
            for i, label in enumerate(self.labels):
                if label != -100:
                    self._prompt_end_in_input_ids = i
                    break
        return self._prompt_end_in_input_ids

    # @property
    # def input_ids(self) -> List[int]:
    #     """
    #     Input ids for the actual generative model. This is a concatination of token_ids and entity_string_token_ids.
    #
    #     :return: input_ids
    #     :rtype: list
    #     """
    #     if self._input_ids is None:
    #         self._generate_input_ids_and_labels()
    #     return self._input_ids
    #
    # @property
    # def labels(self) -> List[int]:
    #     """
    #     Labels for the actual generative model. This is a concatination of [-100] * len(token_ids) and
    #     entity_string_token_ids.
    #
    #     :return: input_ids
    #     :rtype: list
    #     """
    #     if self._labels is None:
    #         self._generate_input_ids_and_labels()
    #     return self._labels

    @property
    def num_input_ids(self) -> int:
        return len(self.input_ids)

    # @property
    # def num_tokens(self) -> int:
    #     return len(self.token_ids)
    #
    # def _generate_input_ids_and_labels(self):
    #     self._input_ids = self.token_ids + self.entity_string_token_ids
    #     self._labels = [-100] * len(self.token_ids) + self.entity_string_token_ids

    @classmethod
    def from_dict(cls, d: Dict):
        d["entities_anno"] = [Entity.from_dict(entity) for entity in d["entities_anno"]]
        return cls(**d)

    def to_dict(self) -> Dict:
        d = self.__dict__
        if self.entities_anno:
            d["entities_anno"] = [entity.to_dict() for entity in self.entities_anno]
        return d


@dataclass
class NERCorpus:
    train: Optional[List[Sentence]] = None
    test: Optional[List[Sentence]] = None
    validation: Optional[List[Sentence]] = None
    name: Optional[str] = None
    entity_tag_to_label: Optional[Dict[int, str]] = None
    _train_len: Optional[int] = None
    _test_len: Optional[int] = None
    _validation_len: Optional[int] = None
    _vocabulary: Optional[NamedEntityVocabulary] = None
    _entity_set: Optional[Set[str]] = None

    @property
    def vocabulary(self) -> NamedEntityVocabulary:
        if not self._vocabulary:
            if self.entity_tag_to_label:
                self._vocabulary = NamedEntityVocabulary.from_entity_tag_to_label_dict(
                    entity_tag_to_label=self.entity_tag_to_label
                )
            else:
                self._vocabulary = NamedEntityVocabulary.from_corpus(corpus=self)
        return self._vocabulary

    @property
    def entity_set(self) -> Set:
        if self._entity_set is None:
            if isinstance(self.vocabulary.entities, set):
                self._entity_set = self.vocabulary.entities
            else:
                self._entity_set = set(
                    [entity.split("-")[1] for entity in self.vocabulary.entities.id2value.values() if len(entity) > 1]
                )
        return self._entity_set

    @property
    def train_len(self) -> int:
        if not self._train_len:
            self._train_len = len(self.train) if self.train else 0
        return self._train_len

    @property
    def test_len(self) -> int:
        if not self._test_len:
            self._test_len = len(self.test) if self.test else 0
        return self._test_len

    @property
    def validation_len(self) -> int:
        if not self._validation_len:
            self._validation_len = len(self.validation) if self.validation else 0
        return self._validation_len

    def __len__(self):
        return self.train_len + self.test_len + self.validation_len

    @property
    def sentences(self) -> Iterator:
        """
        Order: Train set, validation set, test set
        """
        for sentence in self.train + self.validation + self.test:
            yield sentence

    def __getitem__(self, idx: int) -> Sentence:
        """
        This loops through the complete corpus, use with care! No distinction between splits is possible then.

        Order: Train set, validation set, test set
        """
        if idx < self.train_len:
            return self.train[idx]
        elif idx < self.train_len + self._validation_len:
            return self.validation[idx - self.train_len]
        elif idx < self.train_len + self._validation_len + self.test_len:
            return self.validation[idx - self.train_len - self.validation_len]
        else:
            raise IndexError

    def __iter__(self):
        """
        This loops through the complete corpus, use with care! No distinction between splits is possible then.

        Order: Train set, validation set, test set
        """
        for i in range(self.__len__()):
            yield self.__getitem__(idx=i)

    @classmethod
    def from_dict(cls, d: Dict):
        d["train"] = [Sentence.from_dict(sentence) for sentence in d["train"]]
        d["test"] = [Sentence.from_dict(sentence) for sentence in d["test"]]
        d["validation"] = [Sentence.from_dict(sentence) for sentence in d["validation"]]
        return cls(**d)

    def to_dict(self) -> Dict:
        d = self.__dict__
        if d.get("_vocabulary", False):
            # delete vocabulary hidden variable to save space (can easily be recalculated)
            del d["_vocabulary"]
        if self.train:
            d["train"] = [sentence.to_dict() for sentence in self.train]
        if self.test:
            d["test"] = [sentence.to_dict() for sentence in self.test]
        if self.validation:
            d["validation"] = [sentence.to_dict() for sentence in self.validation]
        return d
