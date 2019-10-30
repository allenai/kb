import os
import logging
import numpy as np
import codecs
from typing import Dict, List, Iterable, Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField, ListField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer

from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator, start_token, sep_token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('kg_probe')
class KgProbeReader(DatasetReader):
    """
    This DatasetReader is designed to read in sentences that render information contained in
    knowledge graph triples. Similar to the BertPreTrainingReader, but leverages provided entity
    spans to ensure that entity-related tokens are properly masked out.

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
                 lazy: bool = False) -> None:

        super().__init__(lazy)

        self._tokenizer_and_candidate_generator = tokenizer_and_candidate_generator
        self._label_indexer = {
            "lm_labels": tokenizer_and_candidate_generator._bert_single_id_indexer["tokens"]
        }

    def _read(self, file_path: str):
        with open(cached_path(file_path), 'r') as f:
            for line in f:
                span_text, sentence = line.strip().split('\t')
                span = tuple(int(x) for x in span_text.split())
                yield self.text_to_instance(sentence, span)

    def text_to_instance(self, sentence: str, span: Tuple[int, ...]):
        token_candidates = self._tokenizer_and_candidate_generator.tokenize_and_generate_candidates(sentence)

        # NOTE: Skipping the padding here since sentences are all quite short.
        vocab = self._tokenizer_and_candidate_generator.bert_tokenizer.vocab
        lm_label_ids = TextField(
            [Token(t, text_id=vocab[t]) for t in token_candidates['tokens']],
            token_indexers=self._label_indexer
        )

        # We need to offset the start and end of the span so that it aligns with word pieces.
        if span[0] == 0:
            start = 1  # Since 0'th elt. is <CLS>
        else:
            start = token_candidates['offsets_a'][span[0] - 1]
        end = token_candidates['offsets_a'][span[1]]

        masked_tokens: List[str] = token_candidates['tokens'].copy()
        mask_indicator = np.zeros(len(masked_tokens), dtype=np.uint8)
        for i in range(start, end):
            masked_tokens[i] = '[MASK]'
            mask_indicator[i] = 1

        token_candidates['tokens'] = masked_tokens

        # mask out the entity candidates
        candidates = token_candidates['candidates']
        for candidate_key in candidates.keys():
            indices_to_mask = []
            for k, candidate_span in enumerate(candidates[candidate_key]['candidate_spans']):
                # (end-1) as candidate spans are exclusive (e.g. candidate_span = (0, 0) has start=0, end=1)
                if (candidate_span[0] >= start and candidate_span[0] <= end-1) or (
                    candidate_span[1] >= start and candidate_span[1] <= end-1):
                    indices_to_mask.append(k)
            for ind in indices_to_mask:
                candidates[candidate_key]['candidate_entities'][ind] = ['@@MASK@@']
                candidates[candidate_key]['candidate_entity_priors'][ind] = [1.0]

        fields = self._tokenizer_and_candidate_generator. \
            convert_tokens_candidates_to_fields(token_candidates)

        fields['lm_label_ids'] = lm_label_ids
        fields['mask_indicator'] = ArrayField(mask_indicator, dtype=np.uint8)

        return Instance(fields)
