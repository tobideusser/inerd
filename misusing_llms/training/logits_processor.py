import logging
from copy import deepcopy
from typing import List, Tuple

import torch
from torch import LongTensor, FloatTensor, BoolTensor
from transformers import LogitsProcessor, PreTrainedTokenizer

from misusing_llms.utils import rindex


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
        leading_space: bool = True,
    ):
        self.tokeniser = tokeniser
        self.entity_type_tokens = entity_type_tokens
        self.combine_token = combine_token
        self.entity_separator_token = entity_separator_token
        self.type_content_separator_token = type_content_separator_token
        self.vocab_size = vocab_size
        self.mask_value = mask_value
        self.leading_space = leading_space

        # apparently, the string ". \n" gets tokenised as a *single* token. This behaviour has been observed from the
        # following tokenisers:
        #   - bigscience/bloom
        # Therefore, we add the token id for this to self.combine_token_ids
        self.combine_token_ids = self._tokenise_with_edge_cases(text=self.combine_token)
        if len(self.tokeniser(text=". " + self.combine_token, add_special_tokens=False).input_ids) == 1:
            self.combine_token_ids.append(
                self.tokeniser(text=". " + self.combine_token, add_special_tokens=False).input_ids
            )
        # the same thing happens for the string " \n " when using the following tokenisers:
        #   - togethercomputer/RedPajama
        #   - tiiuae/falcon
        if len(self.tokeniser(text=" " + self.combine_token + " ", add_special_tokens=False).input_ids) == 1:
            self.combine_token_ids.append(
                self.tokeniser(text=" " + self.combine_token + " ", add_special_tokens=False).input_ids
            )

        self.entity_separator_token_ids = self._tokenise_with_edge_cases(text=self.entity_separator_token)

        self.type_content_separator_token_ids = self._tokenise_with_edge_cases(text=self.type_content_separator_token)
        self.entity_type_token_ids = [self._tokenise_with_edge_cases(text=ett) for ett in self.entity_type_tokens]

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
                self.masks_rule2[token_id] = mask

        self.mask_rule1 = torch.ones(self.vocab_size, dtype=torch.bool)
        for token_ids in self.entity_type_token_ids:
            for token_id in token_ids:
                self.mask_rule1[token_id] = False
            # self.mask_rule1[token_ids[0][0]] = False
            # self.mask_rule1[token_ids[1][0]] = False
        self.mask_rule1[tokeniser.eos_token_id] = False

        self.mask_rule3 = torch.ones(self.vocab_size, dtype=torch.bool)
        for token_ids in self.type_content_separator_token_ids:
            for token_id in token_ids:
                self.mask_rule3[token_id] = False
        # self.mask_rule3[self.type_content_separator_token_ids[0][0]] = False
        # self.mask_rule3[self.type_content_separator_token_ids[1][0]] = False

        # this mask is incomplete, as it also requires the previous token id
        self.mask_rule4b_incomplete = torch.ones(self.vocab_size, dtype=torch.bool)
        for token_ids in self.entity_separator_token_ids:
            for token_id in token_ids:
                self.mask_rule4b_incomplete[token_id] = False
        # self.mask_rule4b_incomplete[self.entity_separator_token_ids[0][0]] = False
        # self.mask_rule4b_incomplete[self.entity_separator_token_ids[1][0]] = False

        # to store the position of the previously predicted token
        # self.rule4_memory = [[-1]] * batch_size

        self.rule4_next_token_memory = [[-1]] * batch_size
        common_special_characters = [".", ",", "'", '"', "-", "#", "$", "%", "&", "(", ")", "*", "+", "/", "!", "?"]
        self.common_special_characters_ids = [
            char[0] for char in self.tokeniser(common_special_characters, add_special_tokens=False).input_ids
        ]

        # self.rule4_edge_case_flag = [False] * batch_size

    def _find_token_id_in_entity_type_list(self, token_id: int) -> Tuple[int, int, int]:
        for i, entity_type in enumerate(self.entity_type_token_ids):
            for ii, entity_type_token_ids in enumerate(entity_type):
                for iii, entity_type_token_id in enumerate(entity_type_token_ids):
                    if token_id == entity_type_token_id:
                        return i, ii, iii
        raise AssertionError(f"token_id {token_id} not in self.entity_type_token_ids")

    def _tokenise_with_edge_cases(self, text: str) -> List[List[int]]:
        """tokenises the str input, once without a leading space and once with leading space"""
        without_leading_space = self.tokeniser(text=text, add_special_tokens=False).input_ids
        with_leading_space = self.tokeniser(text=" " + text, add_special_tokens=False).input_ids
        if not text.isalnum():
            minus_leading_space = [self.tokeniser(text="a" + text, add_special_tokens=False).input_ids[-1]]
        else:
            minus_leading_space = None
        if with_leading_space == without_leading_space or (
            without_leading_space[0] in with_leading_space and len(with_leading_space) > 1
        ):
            if minus_leading_space is not None and minus_leading_space != without_leading_space:
                return [without_leading_space, minus_leading_space]
            else:
                return [without_leading_space]
        else:
            if minus_leading_space is not None and minus_leading_space != without_leading_space:
                return [without_leading_space, with_leading_space, minus_leading_space]
            else:
                return [without_leading_space, with_leading_space]

    def _get_prompt_ids(self, input_ids: LongTensor) -> Tuple[List[List[int]], List[List[int]]]:
        # create a list of lists containing each token id present in the prompt / input. This is required for rule 4.
        prompt_ids = input_ids.tolist()
        prompt_ids_with_leading_space = [[-1]] * len(prompt_ids)
        for i, input_ids_for_each_object in enumerate(prompt_ids):
            position_in_input_ids = None
            for combine_token_id in self.combine_token_ids:
                if combine_token_id[0] in input_ids_for_each_object:
                    position_in_input_ids = rindex(lst=input_ids_for_each_object, value=combine_token_id[0])
                    break
            if position_in_input_ids is None:
                raise ValueError(
                    f"No combine token id found in the input!\n"
                    f"Sentence:\n{self.tokeniser.decode(prompt_ids[i])}"
                    f"Token ids:\n{prompt_ids[i]}"
                )
            # try:
            #     position_in_input_ids = input_ids_for_each_object.index(self.combine_token_ids[1][0])
            # except ValueError:
            #     try:
            #         position_in_input_ids = input_ids_for_each_object.index(self.combine_token_ids[0][0])
            #     except ValueError:
            #         try:
            #             position_in_input_ids = input_ids_for_each_object.index(self.combine_token_ids[2][0])
            #         except ValueError:
            #             logger.error(
            #                 f"No combine token id found in the input!\n"
            #                 f"Sentence:\n{self.tokeniser.decode(prompt_ids[i])}"
            #                 f"Token ids:\n{prompt_ids[i]}"
            #             )
            #             raise ValueError
            #         except IndexError:
            #             logger.error(
            #                 f"No combine token id found in the input! Maybe this is the special case for '. \\n'?\n"
            #                 f"Sentence:\n{self.tokeniser.decode(prompt_ids[i])}"
            #                 f"Token ids:\n{prompt_ids[i]}"
            #             )
            #             raise IndexError
            prompt_ids[i] = [
                prompt_id
                for prompt_id in prompt_ids[i][:position_in_input_ids]
                if prompt_id != self.tokeniser.pad_token_id
            ]

            if self.leading_space:
                # add leading space to first prompt token id (to make sampling from it (case 4) more natural)
                prompt_ids_with_leading_space[i] = self.tokeniser(" " + self.tokeniser.decode(prompt_ids[i])).input_ids
            else:
                prompt_ids_with_leading_space[i] = prompt_ids[i]

        return prompt_ids, prompt_ids_with_leading_space

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

    def _get_previous_token_ids(self, input_ids) -> List[int]:
        input_ids = input_ids.tolist()
        previous_token_ids = []
        for input_ids_ in input_ids:
            flag = True
            for input_id in reversed(input_ids_):
                if self.tokeniser.decode(input_id) != "":
                    previous_token_ids.append(input_id)
                    flag = False
                    break
            if flag:
                previous_token_ids.append(input_ids_[-1])
        return previous_token_ids

    def _update_rule4_memory(self, text: str, token: str, batch_position: int):
        """
        Updates the class variable self.rule4_next_token_memory to hold possible token ids for the next prediction.
        """

        if token == "" or token == " ":

            self.rule4_next_token_memory[batch_position] = [-1]

        else:
            # first, get what is written after the token
            text_after_predicted_token = self._get_remaining_text_left_for_generation(text=text, token=token)

            if text_after_predicted_token is not None:
                # tokenise this and get the first token
                token_ids_text_after_predicted_token = [
                    token_ids[0]
                    for token_ids in self.tokeniser(text_after_predicted_token, add_special_tokens=False).input_ids
                    if len(token_ids) > 0
                ]

                # check for edge case of special character predicted
                if any(
                    [
                        True if token_id in self.common_special_characters_ids else False
                        for token_id in token_ids_text_after_predicted_token
                    ]
                ):
                    # this is very specific edge case: If the final character of an entity is a special character, e.g.
                    # ".", the model should be allowed to predict ".;", as this is often tokenised as a single token
                    # (for whatever reason...). Therefore, we add this to the allowed token_ids.
                    token_id_to_add = []
                    for token_id in token_ids_text_after_predicted_token:
                        if token_id in self.common_special_characters_ids:
                            token = self.tokeniser.decode(token_id) + ";"
                            token_id_to_add.append(self.tokeniser(token, add_special_tokens=False).input_ids[0])
                    token_ids_text_after_predicted_token.extend(token_id_to_add)

                # only the next token in this sequence is allowed for prediction
                self.rule4_next_token_memory[batch_position] = token_ids_text_after_predicted_token

                # if text_after_predicted_token is None, the token is not in the text (very likely the model predicted
                # ";").

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
        previous_token_ids = self._get_previous_token_ids(input_ids=input_ids)
        previous_tokens = [self.tokeniser.decode(previous_token_id) for previous_token_id in previous_token_ids]

        # save the device the scores are on (again for easier access)
        device = scores.device

        # determine if we are in the entity content generation phase
        case4: List[bool] = []
        for input_ids_ in input_ids.tolist():
            flag = None
            for input_id in reversed(input_ids_):
                input_id_decoded_to_string = self.tokeniser.decode(input_id)
                if (
                    self.entity_separator_token in input_id_decoded_to_string
                    or [input_id] in self.combine_token_ids + self.entity_separator_token_ids
                ):
                    flag = False
                    break
                elif (
                    self.type_content_separator_token in input_id_decoded_to_string
                    or [input_id] in self.type_content_separator_token_ids
                ):
                    flag = True
                    break
            if flag is None:
                raise ValueError(
                    f"No combine token, entity separator token, or content separator token found in input!\n"
                    f"input ids: {input_ids_}\n"
                    f"decoded: {self.tokeniser.decode(input_ids_)}\n"
                    f"decoded token by token: "
                    f"{[self.tokeniser.decode(input_id).encode('unicode_escape') for input_id in input_ids_]}"
                )
            case4.append(flag)

        # loop through batch self.tokeniser.decode(int(torch.argmax(scores[i])))
        for i, (previous_token_id, previous_token) in enumerate(zip(previous_token_ids, previous_tokens)):
            predicted_token_id_without_masking = int(torch.argmax(scores[i]))
            predicted_token_without_masking = self.tokeniser.decode(predicted_token_id_without_masking)
            if predicted_token_without_masking == "" or predicted_token_without_masking == " ":
                pass
            elif not case4[i]:
                if [previous_token_id] in self.combine_token_ids or self.entity_separator_token in previous_token:
                    # case 1:
                    #   After the "combine_token" or the entity separator (";") the first token of an entity type or the
                    #   end of sequence (EOS) token has to be predicted.
                    scores[i].masked_fill_(mask=self.mask_rule1.to(device), value=self.mask_value)
                elif previous_token_id in self.entity_type_token_ids_begin_and_mid_flattened:
                    # case 2:
                    #   After predicting the first token of any entity type token, the entity type must be completed
                    #   before any other tokens are predicted.

                    # get tuple from entity type lookup, this tells us which entity type the token belongs to
                    # entity_type_position = self.entity_type_token_ids_begin_and_mid_lookup[
                    #     self.entity_type_token_ids_begin_and_mid_flattened.index(previous_token_id)
                    # ]

                    # apply the mask
                    scores[i].masked_fill_(mask=self.masks_rule2[previous_token_id].to(device), value=self.mask_value)
                elif [previous_token_id] in self.entity_type_token_ids_end:
                    # case 3:
                    #   After predicting the last token of an entity type, the type-content separator (":") has to be
                    #   predicted.

                    if predicted_token_without_masking != self.type_content_separator_token:

                        scores[i].masked_fill_(mask=self.mask_rule3.to(device), value=self.mask_value)
            else:
                prompt_ids, prompt_ids_with_leading_space = self._get_prompt_ids(input_ids=input_ids)

                prompt_decoded = self.tokeniser.decode(prompt_ids_with_leading_space[i])

                if [previous_token_id] in self.type_content_separator_token_ids:
                    # case 4a:
                    #   After the type-content separator (":") any token from the input may be predicted.

                    # predicted_token_without_masking = self.tokeniser.decode(int(torch.argmax(scores[i])))
                    if predicted_token_without_masking in prompt_decoded:
                        # model is already correct, no need for additional masking, all we need to do is to make it
                        # clear what is allowed to be predicted next
                        self._update_rule4_memory(
                            text=prompt_decoded, token=predicted_token_without_masking, batch_position=i
                        )

                    else:
                        mask_rule4a = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
                        mask_rule4a[prompt_ids_with_leading_space[i]] = False

                        # apply the mask
                        scores[i].masked_fill_(mask=mask_rule4a, value=self.mask_value)

                        # get token that was predicted
                        predicted_token = self.tokeniser.decode(int(torch.argmax(scores[i])))

                        # update the rule4 memory based upon this
                        self._update_rule4_memory(text=prompt_decoded, token=predicted_token, batch_position=i)

                        # if predicted_token in [",", ".", " .", " ,"]:
                        #     print("WHAT?! DEBUG HERE! line 170")

                else:
                    # case 4b:
                    #   After a token from the input has been predicted, the only allowed tokens for prediction are
                    #   either the entity separator (";") or the token following the previous token in the input.

                    if self.entity_separator_token not in predicted_token_without_masking:
                        # if self.entity_separator_token in predicted_token_without_masking -> model predicts the entity
                        # to end, so no need for any masking

                        # combine the predicted previously predicted token with currently predicted token
                        predicted_token_sequence = self.tokeniser.decode(
                            [previous_token_id, predicted_token_id_without_masking]
                        )

                        # if this sequence is in the prompt, it is an allowed prediction
                        if (
                            predicted_token_sequence in prompt_decoded
                            or predicted_token_sequence in prompt_decoded.replace(" ", "")
                        ):
                            predicted_token = predicted_token_without_masking

                        else:
                            if self.rule4_next_token_memory[i] != [-1]:

                                # deepcopy "incomplete" mask for rule 4b
                                mask_rule4b = deepcopy(self.mask_rule4b_incomplete)

                                # "complete" the mask by adding the next token ids
                                mask_rule4b[self.rule4_next_token_memory[i]] = False

                                # apply the mask
                                scores[i].masked_fill_(mask=mask_rule4b.to(device), value=self.mask_value)

                            # get token that was predicted with masking applied
                            predicted_token = self.tokeniser.decode(int(torch.argmax(scores[i])))

                        # update the rule4 memory based upon the result of this step
                        self._update_rule4_memory(text=prompt_decoded, token=predicted_token, batch_position=i)

            # predicted_token_id = int(torch.argmax(scores[i]))
            # predicted_token = self.tokeniser.decode(predicted_token_id)
            # if ("," in predicted_token or "." in predicted_token) and self.tokeniser.decode(input_ids[i])[-1] == ":":
            #     logger.warning(
            #         f". or , in predicted_token\n"
            #         f"input_ids: {self.tokeniser.decode(input_ids[i])}\n"
            #         f"predicted_token: {predicted_token}"
            #     )
            #     print("WHAT?! DEBUG HERE! line 317")
        return scores
