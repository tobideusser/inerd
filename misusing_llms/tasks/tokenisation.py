import logging
from typing import Dict, Optional, Union

from fluidml.common import Task
from tqdm import tqdm
from transformers import AutoTokenizer

from misusing_llms.data_classes import NERCorpus
from misusing_llms.utils.utils import set_seed_number, set_seeds

logger = logging.getLogger(__name__)


class Tokenisation(Task):
    def __init__(
        self,
        special_tokens: Optional[Dict] = None,
        seed: int = 3141,
        tokeniser_name: Optional[str] = None,
        train_mode: bool = True,
    ):
        super().__init__()

        # config params
        self.special_tokens = special_tokens
        if "facebook/opt" in tokeniser_name:
            logger.info("Using GPT-2 *fast* tokeniser instead of the default slow tokeniser specified for OPT models.")
            self.tokeniser_name = "gpt2"
        else:
            self.tokeniser_name = tokeniser_name
        self.seed = seed

        self.train_mode = train_mode

        self.tokeniser = AutoTokenizer.from_pretrained(self.tokeniser_name, use_fast=True)
        if self.special_tokens is not None:
            self.tokeniser.add_special_tokens(self.special_tokens)

    def _tokenise_corpus(self, corpus: NERCorpus) -> NERCorpus:

        for sentence in tqdm(corpus.sentences):
            sentence.token_ids = self.tokeniser(sentence.content).input_ids
            sentence.tokens = self.tokeniser.convert_ids_to_tokens(sentence.token_ids)

            sentence.entity_string_token_ids = self.tokeniser(sentence.entity_string).input_ids
            sentence.entity_string_tokens = self.tokeniser.convert_ids_to_tokens(sentence.entity_string_token_ids)

        return corpus

    def run(self, corpus_parsed: Union[Dict, NERCorpus]):
        set_seed_number(self.seed)
        set_seeds()

        if isinstance(corpus_parsed, Dict):
            logger.info("Converting corpus_parsed dict to Corpus object.")
            corpus = NERCorpus.from_dict(corpus_parsed)
        else:
            corpus = corpus_parsed

        logger.debug("Tokenise corpus...")
        corpus_tokenised = self._tokenise_corpus(corpus)

        if self.train_mode:
            self.save(corpus_tokenised.to_dict(), "corpus_tokenised", type_="pickle")
            self.save(self.tokeniser, "tokeniser", type_="tokeniser")
        else:
            return corpus_tokenised, self.tokeniser
