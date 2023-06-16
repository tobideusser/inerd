import logging
from typing import Dict, Optional, Union, List

from fluidml import Task
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizer

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
        add_leading_space: bool = True,
        llama: bool = False,
    ):
        super().__init__()

        # config params
        self.combine_token = combine_token
        self.special_tokens = special_tokens
        self.tokeniser_name = tokeniser_name
        self.add_leading_space = add_leading_space
        self.seed = seed

        self.train_mode = train_mode

        if llama:
            self.tokeniser = LlamaTokenizer.from_pretrained(self.tokeniser_name)
        else:
            self.tokeniser = AutoTokenizer.from_pretrained(self.tokeniser_name)

    def _tokenise_corpus(self, corpus: NERCorpus) -> NERCorpus:

        for sentence in tqdm(corpus.sentences, total=len(corpus)):
            if self.add_leading_space:
                prompt_tokens = " " + sentence.content + " " + self.combine_token
            else:
                prompt_tokens = sentence.content + " " + self.combine_token
            input_tokens = prompt_tokens + " " + sentence.entity_string + self.tokeniser.eos_token

            sentence.input_ids = self.tokeniser(input_tokens).input_ids
            sentence.input_tokens = self.tokeniser.batch_decode(sentence.input_ids)

            length_tokenised_prompt = len(self.tokeniser(prompt_tokens).input_ids)
            sentence.labels = [-100] * length_tokenised_prompt + sentence.input_ids[length_tokenised_prompt:]

        return corpus

    def run(self, corpus_parsed: Union[Dict, NERCorpus, List]):
        set_seed_number(self.seed)
        set_seeds()
        self.unique_config["Parsing"].get("combine_token", "\n")

        if isinstance(corpus_parsed, Dict):
            logger.info("Converting corpus_parsed dict to NERCorpus object.")
            corpus = NERCorpus.from_dict(corpus_parsed)
            entity_separator_token = corpus[0].entity_separator_token
            type_content_separator_token = corpus[0].type_content_separator_token
        elif isinstance(corpus_parsed, list):
            logger.info("Converting corpus_parsed list of dict to list of NERCorpus object.")
            corpus = [NERCorpus.from_dict(c) for c in corpus_parsed]
            entity_separator_token = corpus[0][0].entity_separator_token
            type_content_separator_token = corpus[0][0].type_content_separator_token
        else:
            corpus = corpus_parsed
            entity_separator_token = corpus[0].entity_separator_token
            type_content_separator_token = corpus[0].type_content_separator_token

        self.tokeniser.add_special_tokens(
            {
                "additional_special_tokens": [entity_separator_token, type_content_separator_token],
            }
        )

        logger.info("Tokenise corpus...")
        if isinstance(corpus, list):
            corpus_tokenised = []
            for c in corpus:
                logger.info(f"Tokenising {c.name}")
                corpus_tokenised.append(self._tokenise_corpus(c))
        else:
            corpus_tokenised = self._tokenise_corpus(corpus)

        if self.train_mode:
            if isinstance(corpus, list):
                self.save([c.to_dict() for c in corpus], "corpus_tokenised", type_="pickle")
            else:
                self.save(corpus.to_dict(), "corpus_tokenised", type_="pickle")
        else:
            return corpus_tokenised, self.tokeniser
