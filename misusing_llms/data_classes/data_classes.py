from dataclasses import dataclass
from typing import List, Dict, Optional, Iterator, Union

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
    entities: Index  # for now: IOB tags, might have to be adjusted when using non IOB methods

    @classmethod
    def from_entity_tag_to_label_dict(cls, entity_tag_to_label: Dict):
        return cls(entities=Index(id2value=entity_tag_to_label))

    @classmethod
    def from_corpus(cls, corpus):
        raise NotImplementedError


@dataclass
class Sentence:
    words: List[str]
    entity_tags: List[int]
    id_: int
    entity_label: Optional[List[str]] = None

    token_ids: Optional[List[int]] = None
    tokens: Optional[List[str]] = None

    word2token_alignment_mask: Optional[Union[List[List[bool]], Tensor]] = None
    word2token_start_ids: Optional[List[int]] = None
    word2token_end_ids: Optional[List[int]] = None

    entity_iobes: Optional[List[str]] = None

    wimp_score: Optional[List[float]] = None
    wimp_attention_score: Optional[List[float]] = None
    wimp_mlm_score: Optional[List[float]] = None
    wimp_attention_score_type: Optional[str] = None
    _content: Optional[str] = None

    def __len__(self):
        return len(self.words)

    @property
    def content(self) -> str:
        if not self._content:
            self._content = " ".join(self.words)
        return self._content

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)

    def to_dict(self) -> Dict:
        return self.__dict__


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
