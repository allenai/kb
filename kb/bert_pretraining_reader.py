import os
import logging
import numpy as np
import codecs
from typing import Dict, List, Iterable, Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator, start_token, sep_token

from itertools import chain

import random

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def mask_entities(lm_labels, all_candidate_spans):
    """
    lm_labels = [PAD] where not making a prediction, otherwise the target token
    all_candidate_spans = list of span start/end 

    returns spans_to_mask, spans_to_random
        each is a list of span start/end
        spans_to_mask = candidate spans to replace with @@MASK@@
        spans_to_random = candidate_spans to replace with random entities

    For each candidate span that overlaps with a masked location:
        80% we mask out
        10% we keep as is
        10% we replace with random
    """
    masked_indices = [index for index, lm_label in enumerate(lm_labels)
                      if lm_label != '[PAD]']

    spans_to_mask = set()
    spans_to_random = set()
    for index in masked_indices:
        for span in all_candidate_spans:
            if index >= span[0] and index <= span[1]:
                # this candidate has been masked
                if np.random.random() < 0.8:
                    # mask it out
                    spans_to_mask.add(tuple(span))
                else:
                    if np.random.random() < 0.5:
                        # keep as is, do nothing
                        pass
                    else:
                        # random
                        spans_to_random.add(tuple(span))

    return spans_to_mask, spans_to_random


def replace_candidates_with_mask_entity(candidates, spans_to_mask):
    """
    candidates = key -> {'candidate_spans': ...}
    """
    for candidate_key in candidates.keys():
        indices_to_mask = []
        for k, candidate_span in enumerate(candidates[candidate_key]['candidate_spans']):
            if tuple(candidate_span) in spans_to_mask:
                indices_to_mask.append(k)
        for ind in indices_to_mask:
            candidates[candidate_key]['candidate_entities'][ind] = ['@@MASK@@']
            candidates[candidate_key]['candidate_entity_priors'][ind] = [1.0]


def replace_candidates_with_random_entity(candidates, spans_to_random):
    for candidate_key in candidates.keys():

        all_entities = list(set(chain.from_iterable(candidates[candidate_key]['candidate_entities'])))
        
        indices_to_random = []
        for k, candidate_span in enumerate(candidates[candidate_key]['candidate_spans']):
            if tuple(candidate_span) in spans_to_random:
                indices_to_random.append(k)

        for ind in indices_to_random:
            n = np.random.randint(5) + 1
            random.shuffle(all_entities)
            rand_entities = all_entities[:n]
            candidates[candidate_key]['candidate_entities'][ind] = list(rand_entities)
            prior = np.random.rand(len(rand_entities))
            prior /= prior.sum()
            candidates[candidate_key]['candidate_entity_priors'][ind] = prior.tolist()


class BertTokenizerCandidateGeneratorMasker:
    """
    Handles:
        * tokenizing sentence_a, sentence_b
        * generating candidates
        * adjust candidate spans for word pieces
        * LM masking
        * interaction of LM masking with candidates
        * converting to fields
    """
    def __init__(self,
                 tokenizer_and_candidate_generator: TokenizerAndCandidateGenerator,
                 max_predictions_per_seq: int = 20,
                 masked_lm_prob: float = 0.15,
                 mask_candidate_strategy: str = 'none'):

        self.tokenizer_and_candidate_generator = tokenizer_and_candidate_generator

        self.max_predictions_per_seq = max_predictions_per_seq
        self.masked_lm_prob = masked_lm_prob

        self._label_indexer = {"lm_labels": self.tokenizer_and_candidate_generator._bert_single_id_indexer["tokens"]}
        assert mask_candidate_strategy in ('none', 'full_mask')
        self._mask_candidate_strategy = mask_candidate_strategy

    def tokenize_candidates_mask(self, sentence1: str, sentence2: str):
        """
        # call BertTokenizerAndCandidateGenerator.tokenize_and_generate_candidates
        # call convert_tokens_candidates_to_fields to convert to fields
        # do LM masking, and convert LM masks to fields
        """
        # Generate entity candidates
        token_candidates = self.tokenizer_and_candidate_generator.tokenize_and_generate_candidates(
            sentence1, sentence2
        )

        # LM masking
        masked_tokens, lm_labels = self.create_masked_lm_predictions(
                token_candidates['tokens']
        )

        # masking interaction with spans
        if self._mask_candidate_strategy == 'full_mask':
            all_candidate_spans = []
            for key in token_candidates['candidates'].keys():
                all_candidate_spans.extend(
                        token_candidates['candidates'][key]['candidate_spans']
                )

            spans_to_mask, spans_to_random = mask_entities(lm_labels, all_candidate_spans)
            replace_candidates_with_mask_entity(
                    token_candidates['candidates'], spans_to_mask
            )
            replace_candidates_with_random_entity(
                    token_candidates['candidates'], spans_to_random
            )

        token_candidates['tokens'] = masked_tokens

        # Converting to fields
        fields = self.tokenizer_and_candidate_generator. \
            convert_tokens_candidates_to_fields(token_candidates)

        # Adding LM labels field
        fields['lm_label_ids'] = TextField(
            [Token(t, text_id=self.tokenizer_and_candidate_generator.bert_tokenizer.vocab[t]) for t in lm_labels],
            token_indexers=self._label_indexer
        )

        return fields


    def create_masked_lm_predictions(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        """Creates the predictions for the masked LM objective.
           Assumes tokens is already word piece tokenized and truncated"""

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == start_token or token == sep_token:
                continue
            cand_indexes.append(i)

        np.random.shuffle(cand_indexes)

        # the return list of tokens, with [MASK]
        output_tokens = list(tokens)

        num_to_predict = min(self.max_predictions_per_seq,
            max(1, int(round(len(tokens) * self.masked_lm_prob)))
        )

        lm_labels = ["[PAD]"] * len(tokens)
        vocab = self.tokenizer_and_candidate_generator.bert_tokenizer.ids_to_tokens

        for index in cand_indexes[:num_to_predict]:
            # 80% of the time, replace with [MASK]
            if np.random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if np.random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab[np.random.randint(0, len(vocab))]

            output_tokens[index] = masked_token

            lm_labels[index] = tokens[index]

        return output_tokens, lm_labels



@DatasetReader.register("bert_pre_training")
class BertPreTrainingReader(DatasetReader):
    """
    This DatasetReader is designed to read in text corpora segmented into sentences and yield
    documents for training BERT models.

    It returns a dataset of instances with the following fields:

    tokens : ``TextField``
        The WordPiece tokens in the sentence.
    segment_ids : ``SequenceLabelField``
        The labels of each of the tokens (0 - tokens from the first sentence,
        1 - tokens from the second sentence).
    lm_label_ids : ``SequenceLabelField``
        For each masked position, what is the correct label.
    next_sentence_label : ``LabelField``
        Next sentence label: is the second sentence the next sentence following the
        first one, or is it a randomly selected sentence.
    candidates: ``DictField``
    """

    def __init__(self,
                 tokenizer_and_candidate_generator: TokenizerAndCandidateGenerator,
                 max_predictions_per_seq: int = 20,
                 masked_lm_prob: float = 0.15,
                 mask_candidate_strategy: str = 'none',
                 lazy: bool = False) -> None:

        super().__init__(lazy)

        self._tokenizer_masker = BertTokenizerCandidateGeneratorMasker(
                 tokenizer_and_candidate_generator,
                 max_predictions_per_seq=max_predictions_per_seq,
                 masked_lm_prob=masked_lm_prob,
                 mask_candidate_strategy=mask_candidate_strategy)


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        with open(cached_path(file_path), 'r') as fin:
            for line in fin:
                label, sentence1, sentence2 = line.strip().split('\t')
                yield self.text_to_instance(sentence1, sentence2, int(label))

    def text_to_instance(self,
                        sentence1: str,
                        sentence2: str,
                        next_sentence_label: int):

        fields = self._tokenizer_masker.tokenize_candidates_mask(sentence1, sentence2)

        # NSP label field
        fields['next_sentence_label'] = \
            LabelField(next_sentence_label, skip_indexing=True)

        return Instance(fields)

