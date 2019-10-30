import json
import logging
from typing import Dict, Iterable, List, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.common.registrable import Registrable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np

from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator
from kb.common import MentionGenerator

logger = logging.getLogger(__name__)


LABEL_MAP = {
    'no_relation': 0,
    'org:alternate_names': 1,
    'org:city_of_headquarters': 2,
    'org:country_of_headquarters': 3,
    'org:dissolved': 4,
    'org:founded': 5,
    'org:founded_by': 6,
    'org:member_of': 7,
    'org:members': 8,
    'org:number_of_employees/members': 9,
    'org:parents': 10,
    'org:political/religious_affiliation': 11,
    'org:shareholders': 12,
    'org:stateorprovince_of_headquarters': 13,
    'org:subsidiaries': 14,
    'org:top_members/employees': 15,
    'org:website': 16,
    'per:age': 17,
    'per:alternate_names': 18,
    'per:cause_of_death': 19,
    'per:charges': 20,
    'per:children': 21,
    'per:cities_of_residence': 22,
    'per:city_of_birth': 23,
    'per:city_of_death': 24,
    'per:countries_of_residence': 25,
    'per:country_of_birth': 26,
    'per:country_of_death': 27,
    'per:date_of_birth': 28,
    'per:date_of_death': 29,
    'per:employee_of': 30,
    'per:origin': 31,
    'per:other_family': 32,
    'per:parents': 33,
    'per:religion': 34,
    'per:schools_attended': 35,
    'per:siblings': 36,
    'per:spouse': 37,
    'per:stateorprovince_of_birth': 38,
    'per:stateorprovince_of_death': 39,
    'per:stateorprovinces_of_residence': 40,
    'per:title': 41
}


@DatasetReader.register('tacred')
class TacredDatasetReader(DatasetReader):
    """Reads TACRED data.

    Parameters:
        tokenizer_and_candidate_generator : ``TokenizerAndCandidateGenerator``
            Used to tokenize text, and obtain linking candidates.
        entity_masking : ``str``, optional
            Entity masking strategy (see Section 3.3 of https://openreview.net/pdf?id=BJgrxbqp67)
            that replaces entity tokens with special mask tokens. One of:
                "mask" : Entity mentions are replaced with the "[MASK]" token.
                "type/role" : Entity mentions are replaced with their type and grammatical role.
                "type/role/segment": entity mentions are unmasked, their types are appended to the input, and segment ids are used
                "entity_markers": the entity markers approach from Soares et al ACL 2019 - entities are unmasked, no types are used, marked with [e1start] [e1end], [e2start], [e2end]
                "entity_markers/type": the entity markers approach followed by [SEP] subj type [SEP] obj type
            By default no masking is used.
    """
    def __init__(self,
                 tokenizer_and_candidate_generator: TokenizerAndCandidateGenerator,
                 entity_masking: str = None,
                 lazy: bool = False,) -> None:
        super().__init__(lazy=lazy)
        if entity_masking is not None:
            assert entity_masking in [
                'mask',
                'type/role',
                'type/role/segment',
                'entity_markers',
                'entity_markers/type'
        ]
        self.tokenizer_and_candidate_generator = tokenizer_and_candidate_generator
        self.tokenizer_and_candidate_generator.whitespace_tokenize = True
        self.entity_masking = entity_masking

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), 'r') as f:
            tacred_data = json.load(f)

            for example in tacred_data:
                tokens = example['token']
                relation = example['relation']

                subj_start = example['subj_start']
                subj_end = example['subj_end']
                subj_type = example['subj_type']
                subj_tokens = tokens[subj_start:subj_end+1]

                obj_start = example['obj_start']
                obj_end = example['obj_end']
                obj_type = example['obj_type']
                obj_tokens = tokens[obj_start:obj_end+1]

                def mask(x, i):
                    if self.entity_masking is None:
                        return x
                    if self.entity_masking == 'type/role/segment':
                        return x
                    if self.entity_masking in ('entity_markers', 'entity_markers/type'):
                        return x
                    if i == subj_start:
                        if self.entity_masking == 'mask':
                            return '[MASK]'
                        else:
                            return '[s-%s]' % subj_type.lower()
                    elif subj_start < i <= subj_end:
                            return None
                    elif i == obj_start:
                        if self.entity_masking == 'mask':
                            return '[MASK]'
                        else:
                            return '[o-%s]' % obj_type.lower()
                    elif obj_start < i <= obj_end:
                        return None
                    else:
                        return x

                masked_tokens = [mask(x, i) for i, x in enumerate(tokens) if mask(x, i) is not None]
                if self.entity_masking == 'type/role/segment':
                    all_tokens = masked_tokens + [
                                '[SEP]',
                                '[s-%s]' % subj_type.lower(),
                                '[SEP]',
                                '[o-%s]' % obj_type.lower()
                    ]
                elif self.entity_masking in ('entity_markers', 'entity_markers/type'):
                    all_tokens = list(masked_tokens)

                    all_tokens.insert(subj_end+1, '[e1end]')
                    all_tokens.insert(subj_start, '[e1start]')

                    subj_end += 2

                    if subj_start < obj_start:
                        obj_start += 2
                        obj_end += 2

                    all_tokens.insert(obj_end+1, '[e2end]')
                    all_tokens.insert(obj_start, '[e2start]')

                    if obj_start < subj_start:
                        subj_start += 2
                        subj_end += 2

                    obj_end += 2

                    if self.entity_masking == 'entity_markers/type':
                        all_tokens.extend([
                            '[SEP]',
                            '[s-%s]' % subj_type.lower(),
                            '[SEP]',
                            '[o-%s]' % obj_type.lower()
                        ])

                else:
                    all_tokens = subj_tokens + ['[SEP]'] + obj_tokens + ['[SEP]'] + masked_tokens

                sentence = ' '.join(all_tokens)

                yield self.text_to_instance(sentence, relation,
                                            subj_start, subj_end,
                                            obj_start, obj_end)

    def text_to_instance(self,
                         sentence: Iterable[str],
                         relation: str,
                         subj_start: int,
                         subj_end: int,
                         obj_start: int,
                         obj_end: int) -> Instance:
        """
        Following approach in:
            https://openreview.net/forum?id=BJgrxbqp67
        We modify the input to look like:
            [CLS] subj [SEP] obj [SEP] sentence [SEP]
        """
        token_candidates = self.tokenizer_and_candidate_generator.tokenize_and_generate_candidates(sentence)

        if self.entity_masking == 'type/role/segment':
            offsets = [1] + token_candidates['offsets_a'][:-1]
            segment_ids = list(token_candidates['segment_ids'])
            for s, e, ii in [[subj_start, subj_end+1, 1], [obj_start, obj_end+1, 2]]:
                ll = offsets[e] - offsets[s]
                segment_ids[offsets[s]:offsets[e]] = [ii] * ll
            # the type + [SEP]
            segment_ids[-4:-2] = [1, 1]
            segment_ids[-2:] = [2, 2]
            token_candidates['segment_ids'] = segment_ids

        # get the indices of the entity starts
        offsets = [1] + token_candidates['offsets_a'][:-1]
        idx1_offset = offsets[subj_start]
        idx2_offset = offsets[obj_start]

        fields = self.tokenizer_and_candidate_generator.convert_tokens_candidates_to_fields(token_candidates)
        fields['label_ids'] = LabelField(LABEL_MAP[relation], skip_indexing=True)
        fields['index_a'] = LabelField(idx1_offset, skip_indexing=True)
        fields['index_b'] = LabelField(idx2_offset, skip_indexing=True)

        return Instance(fields)
