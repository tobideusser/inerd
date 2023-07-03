import logging
import os
from typing import Optional, Dict, Union

from fluidml import Task

from misusing_llms.data_classes import NERCorpus
from misusing_llms.parser import BaseParser

logger = logging.getLogger(__name__)


class Parsing(Task):
    def __init__(
        self,
        dataset: Union[str, Dict[str, str]],
        entity_separator_token: str,
        type_content_separator_token: str,
        debug_size: Optional[int] = None,
        type_mapping: Optional[Dict] = None,
        train_mode: bool = True,
        **kwargs,
    ):
        super().__init__()

        # config params
        self.dataset_name = kwargs.get("dataset_name", dataset)
        self.debug_size = debug_size
        self.dataset = dataset
        self.type_mapping = type_mapping
        self.entity_separator_token = entity_separator_token
        self.type_content_separator_token = type_content_separator_token

        self.parsing_cfg = kwargs

        self.train_mode = train_mode

    def _parse_dataset(self, dataset: str, path_to_data_folders: Optional[str] = None) -> NERCorpus:
        logger.info(f"Parse {dataset} dataset with debug size {self.debug_size}.")
        parser = BaseParser.load_parser(
            type_=dataset,
            debug_size=self.debug_size,
            type_mapping=self.type_mapping,
            path_to_data_folders=path_to_data_folders,
            type_content_separator_token=self.type_content_separator_token,
            entity_separator_token=self.entity_separator_token,
            cache_dir=os.path.join(self.results_store.base_dir, ".hfcache"),
            **self.parsing_cfg,
        )
        corpus = parser.parse()
        return corpus

    def run(self):
        if isinstance(self.dataset, dict):
            corpus = []
            for dataset, value_field in self.dataset.items():
                if isinstance(value_field, str):
                    if os.path.isdir(value_field):
                        corpus.append(self._parse_dataset(dataset=dataset, path_to_data_folders=value_field))
                    else:
                        raise ValueError(f"{value_field} path of key '{dataset}' does not exist.")
                elif isinstance(value_field, bool):
                    if value_field:
                        corpus.append(self._parse_dataset(dataset=dataset))
                else:
                    raise ValueError(
                        f"Wrong datatype specified: value field of '{dataset}' is of type {type(value_field)}."
                    )

        else:
            corpus = self._parse_dataset(dataset=self.dataset)

        if self.train_mode:
            if isinstance(corpus, list):
                self.save([c.to_dict() for c in corpus], "corpus_parsed", type_="pickle")
            else:
                self.save(corpus.to_dict(), "corpus_parsed", type_="pickle")
        else:
            return corpus
