import logging
from copy import deepcopy
from typing import List, Tuple

import torch
from torch import LongTensor, FloatTensor, BoolTensor
from transformers import LogitsProcessor, PreTrainedTokenizer


logger = logging.getLogger(__name__)


class InformedNERDecoderLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokeniser: PreTrainedTokenizer,
        entity_type_tokens: List[str],
        vocab_size: int,
        combine_token: str,
        entity_separator_token: str,
        type_content_separator_token: str,
        batch_size: int,
        mask_value: int = -1000,
    ):
        self.tokeniser = tokeniser
        self.entity_type_tokens = entity_type_tokens
        self.combine_token = combine_token
        self.entity_separator_token = entity_separator_token
        self.type_content_separator_token = type_content_separator_token
        self.vocab_size = vocab_size
        self.mask_value = mask_value

        # apparently, the string ". \n" gets tokenised as a *single* token. This behaviour has been observed from the
        # following tokenisers:
        #   - bigscience/bloom
        # Therefore, we add the token id for this to self.combine_token_ids
        self.combine_token_ids = self._tokenise_with_leading_space(text=self.combine_token)
        if "bigscience/bloom" in self.tokeniser.name_or_path:
            self.combine_token_ids.append(self.tokeniser(text=". " + self.combine_token).input_ids)
        # the same thing happens for the string " \n " when using the following tokenisers:
        #   - togethercomputer/RedPajama
        elif "togethercomputer/RedPajama" in self.tokeniser.name_or_path:
            self.combine_token_ids.append(self.tokeniser(text=" " + self.combine_token + " ").input_ids)

        self.entity_separator_token_ids = self._tokenise_with_leading_space(text=self.entity_separator_token)

        self.type_content_separator_token_ids = self._tokenise_with_leading_space(
            text=self.type_content_separator_token
        )
        self.entity_type_token_ids = [self._tokenise_with_leading_space(text=ett) for ett in self.entity_type_tokens]

        # divide all entity tokens in either begin and mid or end, this is to make the calculations for rule 2 & 3
        # easier
        self.entity_type_token_ids_begin_and_mid = []
        self.entity_type_token_ids_end = []
        for entity_type_token_ids in self.entity_type_token_ids:
            for entity_type_token_id in entity_type_token_ids:
                if len(entity_type_token_id) == 1:
                    self.entity_type_token_ids_end.append(entity_type_token_id)
                else:
                    self.entity_type_token_ids_end.append([entity_type_token_id[-1]])
                    self.entity_type_token_ids_begin_and_mid.append(entity_type_token_id[:-1])

        # flatten entity_type_token_ids_begin_and_mid so that we can easily check rule 2
        self.entity_type_token_ids_begin_and_mid_flattened = []
        self.entity_type_token_ids_begin_and_mid_lookup = []  # in which entity_type the result was found
        self.masks_rule2 = dict()
        for i, entity_type in enumerate(self.entity_type_token_ids_begin_and_mid):
            for token_id in entity_type:
                self.entity_type_token_ids_begin_and_mid_flattened.append(token_id)
                # find to which entity type the token_id belongs
                lookup_index = self._find_token_id_in_entity_type_list(token_id=token_id)
                self.entity_type_token_ids_begin_and_mid_lookup.append(lookup_index)
                # get token ids from that entity_type
                entity_type_token_ids = self.entity_type_token_ids[lookup_index[0]][lookup_index[1]]
                # from that list we can get what token id has to be predicted next -> the one that comes after our
                # previous token
                only_allowed_token = entity_type_token_ids[entity_type_token_ids.index(token_id) + 1]
                # create mask and append
                mask = torch.ones(self.vocab_size, dtype=torch.bool)
                mask[only_allowed_token] = False
                self.masks_rule2[lookup_index] = mask

        self.mask_rule1 = torch.ones(self.vocab_size, dtype=torch.bool)
        for token_ids in self.entity_type_token_ids:
            self.mask_rule1[token_ids[0][0]] = False
            self.mask_rule1[token_ids[1][0]] = False
        self.mask_rule1[tokeniser.eos_token_id] = False

        self.mask_rule3 = torch.ones(self.vocab_size, dtype=torch.bool)
        self.mask_rule3[self.type_content_separator_token_ids[0][0]] = False
        self.mask_rule3[self.type_content_separator_token_ids[1][0]] = False

        # this mask is incomplete, as it also requires the previous token id
        self.mask_rule4b_incomplete = torch.ones(self.vocab_size, dtype=torch.bool)
        self.mask_rule4b_incomplete[self.entity_separator_token_ids[0][0]] = False
        self.mask_rule4b_incomplete[self.entity_separator_token_ids[1][0]] = False

        # to store the position of the previously predicted token
        self.rule4_memory = [[-1]] * batch_size

        self.rule4_next_token_memory = [[-1]] * batch_size

        self.rule4_edge_case_flag = [False] * batch_size

    def _find_token_id_in_entity_type_list(self, token_id: int) -> Tuple[int, int]:
        for i, entity_type in enumerate(self.entity_type_token_ids):
            for ii, entity_type_token_id in enumerate(entity_type):
                if token_id in entity_type_token_id:
                    return i, ii

    def _tokenise_with_leading_space(self, text: str) -> List[List[int]]:
        """tokenises the str input, once without a leading space and once with leading space"""
        without_leading_space = self.tokeniser(text=text).input_ids
        with_leading_space = self.tokeniser(text=" " + text).input_ids
        if with_leading_space == without_leading_space:
            return [without_leading_space]
        else:
            return [without_leading_space, with_leading_space]

    def _get_prompt_ids(self, input_ids: LongTensor) -> Tuple[List[List[int]], List[List[int]]]:
        # create a list of lists containing each token id present in the prompt / input. This is required for rule 4.
        prompt_ids = input_ids.tolist()
        prompt_ids_with_leading_space = [[-1]] * 5
        for i, input_ids_for_each_object in enumerate(prompt_ids):
            try:
                position_in_input_ids = input_ids_for_each_object.index(self.combine_token_ids[1][0])
            except ValueError:
                try:
                    position_in_input_ids = input_ids_for_each_object.index(self.combine_token_ids[0][0])
                except ValueError:
                    try:
                        position_in_input_ids = input_ids_for_each_object.index(self.combine_token_ids[2][0])
                    except ValueError:
                        logger.error(
                            f"No combine token id found in the input!\n"
                            f"Sentence:\n{self.tokeniser.decode(prompt_ids[i])}"
                            f"Token ids:\n{prompt_ids[i]}"
                        )
                        raise ValueError
                    except IndexError:
                        logger.error(
                            f"No combine token id found in the input! Maybe this is the special case for '. \\n'?\n"
                            f"Sentence:\n{self.tokeniser.decode(prompt_ids[i])}"
                            f"Token ids:\n{prompt_ids[i]}"
                        )
                        raise IndexError
            prompt_ids[i] = [
                prompt_id
                for prompt_id in prompt_ids[i][:position_in_input_ids]
                if prompt_id != self.tokeniser.pad_token_id
            ]

            # add leading space to first prompt token id (to make sampling from it (case 4) more natural)
            prompt_ids_with_leading_space[i] = deepcopy(prompt_ids[i])
            prompt_ids_with_leading_space[i][0] = self.tokeniser(
                " " + self.tokeniser.decode(prompt_ids[i][0])
            ).input_ids[0]

        return prompt_ids, prompt_ids_with_leading_space

    def _apply_rule4a(
        self, scores: FloatTensor, prompt_ids: List[List[int]], batch_position: int, device: torch.device
    ) -> FloatTensor:
        mask_rule4a = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
        mask_rule4a[prompt_ids[batch_position]] = False

        # apply the mask
        scores[batch_position].masked_fill_(mask=mask_rule4a, value=self.mask_value)

        predicted_token_id = int(torch.argmax(scores[batch_position]))
        predicted_token = self.tokeniser.decode(predicted_token_id)
        if predicted_token in [",", ".", " .", " ,"]:
            print("WHAT?! DEBUG HERE! line 170")

        # save position of predicted token
        self.rule4_memory[batch_position] = [
            i for i, x in enumerate(prompt_ids[batch_position]) if x == predicted_token_id
        ]
        return scores

    def _get_remaining_text_left_for_generation(self, text: str, token: str) -> List[str]:
        """
        Given an input, this returns a list with what is left for the generation in rule 4b.

        Example:
            Text: "According to all known laws of aviation, there is no way a bee should be able to fly."
            Token: "to"
            Returns: [" all known laws of aviation, there is no way a bee should be able to fly.", " fly."}
        """
        splitted = text.split(token, maxsplit=1)
        if len(splitted) > 1:
            splitted = [splitted[1]]
            if token in splitted[0]:
                remaining_text = self._get_remaining_text_left_for_generation(text=splitted[0], token=token)
                splitted.extend(remaining_text)
            return splitted

    def _update_rule4_memory(self, text: str, token: str, batch_position: int):
        """
        Updates the class variable self.rule4_next_token_memory to hold possible token ids for the next prediction.
        """

        # first, get what is written after the token
        text_after_predicted_token = self._get_remaining_text_left_for_generation(text=text, token=token)

        if text_after_predicted_token is not None:
            # tokenise this
            token_ids_text_after_predicted_token = self.tokeniser(text_after_predicted_token).input_ids

            # only the next token in this sequence is allowed for prediction
            self.rule4_next_token_memory[batch_position] = [
                token_ids[0] for token_ids in token_ids_text_after_predicted_token
            ]

            # if text_after_predicted_token is None, the token is not in the text (very likely the model predicted ";").

    def __call__(self, input_ids: LongTensor, scores: FloatTensor):
        """
        Enforced rules for NER generation:
          1.  After the "combine_token" or the entity separator (";") the first token of an entity type or the end of
              sequence (EOS) token has to be predicted.
          2.  After predicting the first token of any entity type token, the entity type must be completed before any
              other tokens are predicted.
          3.  After predicting the last token of an entity type, the type-content separator (":") has to be predicted.
          4.  During the entity content prediction phase, i.e. after ":" and before ";" has been predicted, only
              token_ids present in the input_ids and the entity separator (";") are allowed for prediction. This rule
              is divided into two "sub-rules":
              4a.   After the type-content separator (":") any token from the input may be predicted.
              4b.   After a token from the input has been predicted, the only allowed tokens for prediction are
                    either the entity separator (";") or the token following the previous token in the input.
        """
        # dimension of tensor inputs
        # input_ids: (batch_size x padded sequence length)
        # scores: (batch_size x vocab size)

        # save the previous token for easier access
        previous_token_ids = input_ids[:, -1]

        # save the device the scores are on (again for easier access)
        device = scores.device

        # determine if we are in the entity content generation phase
        case4: List[bool] = []
        for input_ids_ in input_ids.tolist():
            flag = None
            for input_id in reversed(input_ids_):
                if [input_id] in self.combine_token_ids + self.entity_separator_token_ids:
                    flag = False
                    break
                elif [input_id] in self.type_content_separator_token_ids:
                    flag = True
                    break
            if flag is None:
                raise ValueError("No combine token, entity separator token, or content separator token found in input!")
            case4.append(flag)

        # loop through batch
        for i, previous_token_id in enumerate(previous_token_ids.tolist()):
            if not case4[i]:
                if [previous_token_id] in self.combine_token_ids + self.entity_separator_token_ids:
                    # case 1:
                    #   After the "combine_token" or the entity separator (";") the first token of an entity type or the
                    #   end of sequence (EOS) token has to be predicted.
                    scores[i].masked_fill_(mask=self.mask_rule1.to(device), value=self.mask_value)
                elif previous_token_id in self.entity_type_token_ids_begin_and_mid_flattened:
                    # case 2:
                    #   After predicting the first token of any entity type token, the entity type must be completed
                    #   before any other tokens are predicted.

                    # get tuple from entity type lookup, this tells us which entity type the token belongs to
                    entity_type_position = self.entity_type_token_ids_begin_and_mid_lookup[
                        self.entity_type_token_ids_begin_and_mid_flattened.index(previous_token_id)
                    ]

                    # apply the mask
                    scores[i].masked_fill_(
                        mask=self.masks_rule2[entity_type_position].to(device), value=self.mask_value
                    )
                elif [previous_token_id] in self.entity_type_token_ids_end:
                    # case 3:
                    #   After predicting the last token of an entity type, the type-content separator (":") has to be
                    #   predicted.
                    scores[i].masked_fill_(mask=self.mask_rule3.to(device), value=self.mask_value)
            else:
                prompt_ids, prompt_ids_with_leading_space = self._get_prompt_ids(input_ids=input_ids)

                prompt_decoded = self.tokeniser.decode(prompt_ids_with_leading_space[i])

                if [previous_token_id] in self.type_content_separator_token_ids:
                    # case 4a:
                    #   After the type-content separator (":") any token from the input may be predicted.

                    predicted_token_id_without_masking = int(torch.argmax(scores[i]))
                    predicted_token_without_masking = self.tokeniser.decode(predicted_token_id_without_masking)
                    if predicted_token_without_masking in prompt_decoded:
                        # model is already correct, no need for additional masking, all we need to do is to make it
                        # clear what is allowed to be predicted next
                        self._update_rule4_memory(
                            text=prompt_decoded, token=predicted_token_without_masking, batch_position=i
                        )

                        # if predicted_token_id_without_masking in prompt_ids_with_leading_space[i]:
                        #     # this should be the norm, as the prompt id should always be split the same, regardless of
                        #     # leading space or not!
                        #     self.rule4_memory[i] = [
                        #         ii
                        #         for ii, x in enumerate(prompt_ids_with_leading_space[i])
                        #         if x == predicted_token_id_without_masking
                        #     ]
                        # else:
                        #     # BUT, for whatever reason, we sometimes get very weird results from the tokeniser.
                        #     # example: "Duran" is split into "D" "uran", whereas " Duran" is split into " Dur" "an"
                        #     # Therefore, if predicted_token_id_without_masking is not in the prompt, this is very likely
                        #     # the case.
                        #     if predicted_token_id_without_masking != prompt_ids_with_leading_space[0]:
                        #         logger.warning(
                        #             f"Predicted token in prompt but not in prompt ids!\n"
                        #             f"Predicted token: '{predicted_token_without_masking}'\n"
                        #             f"Predicted token id: {predicted_token_id_without_masking}\n"
                        #             f"Prompt: {prompt_decoded}\n"
                        #             f"Prompt ids: {prompt_ids[i]}\n"
                        #             f"Prompt with leading space: "
                        #             f"{self.tokeniser.decode(prompt_ids_with_leading_space[i])}\n"
                        #             f"Prompt with leading space ids: {prompt_ids_with_leading_space[i]}\n"
                        #             f"Second best prediction: "
                        #             f"'{self.tokeniser.decode(int(torch.topk(scores[i], k=2).indices[1]))}'"
                        #         )
                        #         scores = self._apply_rule4a(
                        #             scores=scores,
                        #             prompt_ids=prompt_ids_with_leading_space,
                        #             batch_position=i,
                        #             device=device,
                        #         )
                        #     else:
                        #         self.rule4_edge_case_flag[i] = True
                    else:
                        mask_rule4a = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
                        mask_rule4a[prompt_ids_with_leading_space[i]] = False

                        # apply the mask
                        scores[i].masked_fill_(mask=mask_rule4a, value=self.mask_value)

                        # get token that was predicted
                        predicted_token = self.tokeniser.decode(int(torch.argmax(scores[i])))

                        # update the rule4 memory based upon this
                        self._update_rule4_memory(text=prompt_decoded, token=predicted_token, batch_position=i)

                        if predicted_token in [",", ".", " .", " ,"]:
                            print("WHAT?! DEBUG HERE! line 170")

                else:
                    # case 4b:
                    #   After a token from the input has been predicted, the only allowed tokens for prediction are
                    #   either the entity separator (";") or the token following the previous token in the input.

                    # deepcopy "incomplete" mask for rule 4b
                    mask_rule4b = deepcopy(self.mask_rule4b_incomplete)

                    # "complete" the mask by adding the next token ids
                    mask_rule4b[self.rule4_next_token_memory[i]] = False

                    # apply the mask
                    scores[i].masked_fill_(mask=mask_rule4b.to(device), value=self.mask_value)

                    # get token that was predicted
                    predicted_token = self.tokeniser.decode(int(torch.argmax(scores[i])))

                    # update the rule4 memory based upon this
                    self._update_rule4_memory(text=prompt_decoded, token=predicted_token, batch_position=i)

            predicted_token_id = int(torch.argmax(scores[i]))
            predicted_token = self.tokeniser.decode(predicted_token_id)
            if ("," in predicted_token or "." in predicted_token) and self.tokeniser.decode(input_ids[i])[-1] == ":":
                logger.warning(
                    f". or , in predicted_token\n"
                    f"input_ids: {self.tokeniser.decode(input_ids[i])}\n"
                    f"predicted_token: {predicted_token}"
                )
                print("WHAT?! DEBUG HERE! line 317")
        return scores
