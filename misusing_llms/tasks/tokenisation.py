import logging
from typing import Dict, Optional, Union

from fluidml import Task
from tqdm import tqdm
from transformers import AutoTokenizer

from misusing_llms.data_classes import NERCorpus
from misusing_llms.utils.utils import set_seed_number, set_seeds

logger = logging.getLogger(__name__)


class Tokenisation(Task):
    def __init__(
        self,
        combine_token: str = "|",
        special_tokens: Optional[Dict] = None,
        seed: int = 3141,
        tokeniser_name: Optional[str] = None,
        train_mode: bool = True,
    ):
        super().__init__()

        # config params
        self.combine_token = combine_token
        self.special_tokens = special_tokens
        self.tokeniser_name = tokeniser_name
        self.seed = seed

        self.train_mode = train_mode

        self.tokeniser = AutoTokenizer.from_pretrained(self.tokeniser_name)

    def _tokenise_corpus(self, corpus: NERCorpus) -> NERCorpus:

        for sentence in tqdm(corpus.sentences):
            prompt_tokens = sentence.content + " " + self.combine_token
            input_tokens = prompt_tokens + " " + sentence.entity_string + self.tokeniser.eos_token

            sentence.input_ids = self.tokeniser(input_tokens).input_ids
            sentence.input_tokens = self.tokeniser.batch_decode(sentence.input_ids)

            length_tokenised_prompt = len(self.tokeniser(prompt_tokens).input_ids)
            sentence.labels = [-100] * length_tokenised_prompt + sentence.input_ids[length_tokenised_prompt:]

            # sentence.token_ids = self.tokeniser(sentence.content).input_ids
            # sentence.tokens = self.tokeniser.convert_ids_to_tokens(sentence.token_ids)
            #
            # sentence.entity_string_token_ids = self.tokeniser(sentence.entity_string).input_ids
            # sentence.entity_string_tokens = self.tokeniser.convert_ids_to_tokens(sentence.entity_string_token_ids)
            #
            # sentence.combine_token = self.combine_token
            # sentence.combine_token_id = self.tokeniser(self.combine_token).input_ids

        return corpus

    def run(self, corpus_parsed: Union[Dict, NERCorpus]):
        set_seed_number(self.seed)
        set_seeds()

        if isinstance(corpus_parsed, Dict):
            logger.info("Converting corpus_parsed dict to Corpus object.")
            corpus = NERCorpus.from_dict(corpus_parsed)
        else:
            corpus = corpus_parsed

        logger.info("Tokenise corpus...")
        corpus_tokenised = self._tokenise_corpus(corpus)

        if self.train_mode:
            self.save(corpus_tokenised.to_dict(), "corpus_tokenised", type_="pickle")
            # self.save(self.tokeniser, "tokeniser", type_="tokeniser")
        else:
            return corpus_tokenised  # , self.tokeniser
