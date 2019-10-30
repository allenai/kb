from typing import Iterable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField
from allennlp.data.instance import Instance
from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator
from allennlp.common.file_utils import cached_path


@DatasetReader.register("wic")
class WicDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer_and_candidate_generator: TokenizerAndCandidateGenerator,
                 entity_markers: bool = False):
        super().__init__()
        self.label_to_index = {'T': 1, 'F': 0}
        self.tokenizer = tokenizer_and_candidate_generator
        self.tokenizer.whitespace_tokenize = True
        self.entity_markers = entity_markers

    def text_to_instance(self, line) -> Instance:
        raise NotImplementedError

    def _read(self, file_path: str) -> Iterable[Instance]:
        """Creates examples for the training and dev sets."""

        with open(cached_path(file_path + '.gold.txt'), 'r') as f:
            labels = f.read().split()

        with open(cached_path(file_path + '.data.txt'), 'r') as f:
            sentences = f.read().splitlines()
            assert len(labels) == len(sentences), f'The length of the labels and sentences must match. ' \
                f'Got {len(labels)} and {len(sentences)}.'

            for line, label in zip(sentences, labels):
                tokens = line.split('\t')
                assert len(tokens) == 5, tokens

                text_a = tokens[3]
                text_b = tokens[4]
                if self.entity_markers:
                    # insert entity markers
                    idx1, idx2 = [int(ind) for ind in tokens[2].split('-')]
                    tokens_a = text_a.strip().split()
                    tokens_b = text_b.strip().split()
                    tokens_a.insert(idx1, '[e1start]')
                    tokens_a.insert(idx1 + 2, '[e1end]')
                    tokens_b.insert(idx2, '[e2start]')
                    tokens_b.insert(idx2 + 2, '[e2end]')
                    text_a = ' '.join(tokens_a)
                    text_b = ' '.join(tokens_b)

                token_candidates = self.tokenizer.tokenize_and_generate_candidates(text_a, text_b)
                fields = self.tokenizer.convert_tokens_candidates_to_fields(token_candidates)
                fields['label_ids'] = LabelField(self.label_to_index[label], skip_indexing=True)

                # get the indices of the marked words
                # index in the original tokens
                idx1, idx2 = [int(ind) for ind in tokens[2].split('-')]
                offsets_a = [1] + token_candidates['offsets_a'][:-1]
                idx1_offset = offsets_a[idx1]
                offsets_b = [token_candidates['offsets_a'][-1] + 1] + token_candidates['offsets_b'][:-1]
                idx2_offset = offsets_b[idx2]

                fields['index_a'] = LabelField(idx1_offset, skip_indexing=True)
                fields['index_b'] = LabelField(idx2_offset, skip_indexing=True)

                instance = Instance(fields)

                yield instance
