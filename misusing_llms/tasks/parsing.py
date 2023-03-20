import logging
from typing import Optional, Dict

from fluidml import Task

from misusing_llms.parser import BaseParser

logger = logging.getLogger(__name__)


class Parsing(Task):
    def __init__(
        self,
        dataset: str,
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

        self.parsing_cfg = kwargs

        self.train_mode = train_mode

    def run(self):
        logger.info(f"Parse {self.dataset_name} dataset with debug size {self.debug_size}.")
        parser = BaseParser.load_parser(
            type_=self.dataset, debug_size=self.debug_size, type_mapping=self.type_mapping, **self.parsing_cfg
        )
        corpus = parser.parse()

        if self.train_mode:
            self.save(corpus.to_dict(), "corpus_parsed", type_="pickle")
        else:
            return corpus
