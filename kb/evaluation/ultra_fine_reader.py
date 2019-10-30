import json
import logging
from typing import Iterable

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField
from allennlp.data.instance import Instance
import numpy as np

from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator

logger = logging.getLogger(__name__)


LABEL_MAP = {
    'entity': 0,
    'event': 1,
    'group': 2,
    'location': 3,
    'object': 4,
    'organization': 5,
    'person': 6,
    'place': 7,
    'time': 8
}


@DatasetReader.register('ultra_fine')
class UltraFineReader(DatasetReader):
    """
    Reads coarse grained entity typing data from "Ultra-Fine Entity Typing",
    Choi et al, ACL 2018.

    Reads data format from https://github.com/thunlp/ERNIE

    Encodes as:

    entity_masking = 'entity':
        [CLS] The left context [ENTITY] right context . [SEP] entity name [SEP]
            use [unused0] as the [ENTITY] token

    entity_masking = 'entity_markers':
        [CLS] The left context [e1start] entity name [e1end] right context . [SEP]
    """
    def __init__(self,
                 tokenizer_and_candidate_generator: TokenizerAndCandidateGenerator,
                 entity_masking: str = 'entity',
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.tokenizer_and_candidate_generator = tokenizer_and_candidate_generator
        self.tokenizer_and_candidate_generator.whitespace_tokenize = True
        assert entity_masking in ('entity', 'entity_markers')
        self.entity_masking = entity_masking


    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), 'r') as f:
            data = json.load(f)

        for example in data:
            # whitespace separated
            tokens = example['sent']

            left = tokens[:example['start']]
            span = tokens[example['start']:example['end']]
            right = tokens[example['end']:]

            if self.entity_masking == 'entity':
                sentence = left.strip() + ' [unused0] ' + right.strip()
                span = span.strip()
                index_entity_start = None
            elif self.entity_masking == 'entity_markers':
                sentence = left.strip() + ' [e1start] ' + span.strip() + ' [e1end] ' + right.strip()
                span = None
                index_entity_start = sentence.split().index('[e1start]')

            # get the labels
            labels = [0] * len(LABEL_MAP)
            for label in example['labels']:
                labels[LABEL_MAP[label]] = 1

            yield self.text_to_instance(sentence, span, labels, index_entity_start)

    def text_to_instance(self, sentence, span, labels, index_entity_start):
        token_candidates = self.tokenizer_and_candidate_generator.tokenize_and_generate_candidates(sentence, span)
        fields = self.tokenizer_and_candidate_generator.convert_tokens_candidates_to_fields(token_candidates)
        fields['label_ids'] = ArrayField(np.array(labels), dtype=np.int)

        # index of entity start
        if index_entity_start is not None:
            offsets = [1] + token_candidates['offsets_a'][:-1]
            idx1_offset = offsets[index_entity_start]
            fields['index_a'] = LabelField(idx1_offset, skip_indexing=True)

        return Instance(fields)

