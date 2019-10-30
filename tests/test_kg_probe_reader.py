import unittest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data import DatasetReader
import numpy as np

from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator
from kb.kg_probe_reader import KgProbeReader
from kb.wordnet import WordNetCandidateMentionGenerator


def get_reader():
    params = {
        "type": "kg_probe",
        "tokenizer_and_candidate_generator": {
            "type": "bert_tokenizer_and_candidate_generator",
            "entity_candidate_generators": {
                "wordnet": {"type": "wordnet_mention_generator",
                            "entity_file": "tests/fixtures/wordnet/entities_fixture.jsonl"}
            },
            "entity_indexers":  {
                "wordnet": {
                       "type": "characters_tokenizer",
                       "tokenizer": {
                           "type": "word",
                           "word_splitter": {"type": "just_spaces"},
                       },
                       "namespace": "entity"
                    }
            },
            "bert_model_type": "tests/fixtures/bert/vocab.txt",
            "do_lower_case": True,
        },
    }

    return DatasetReader.from_params(Params(params))


class TestKgProbeReader(unittest.TestCase):
    def test_kg_probe_reader(self):
        reader = get_reader()
        instances = ensure_list(reader.read('tests/fixtures/kg_probe/file1.txt'))

        # Check instances are correct length
        self.assertEqual(len(instances), 2)

        # Check masking is performed properly
        expected_tokens_0 = ['[CLS]', '[MASK]', '[MASK]', '[UNK]', 'quick',
                             '##est', '.', '[SEP]']
        tokens_0 = [x.text for x in instances[0]['tokens'].tokens]
        self.assertListEqual(expected_tokens_0, tokens_0)

        expected_mask_indicator_0 = np.array([0,1,1,0,0,0,0,0], dtype=np.uint8)
        mask_indicator_0 = instances[0]['mask_indicator'].array
        assert np.allclose(expected_mask_indicator_0, mask_indicator_0)

        expected_tokens_1 = ['[CLS]', 'the', 'brown', 'fox', 'jumped', 'over',
                             'the', '[MASK]', '[MASK]', '[MASK]', '[MASK]',
                             '.', '[SEP]']
        tokens_1 = [x.text for x in instances[1]['tokens'].tokens]
        self.assertListEqual(expected_tokens_1, tokens_1)

        expected_mask_indicator_1 = np.array([0,0,0,0,0,0,0,1,1,1,1,0,0], dtype=np.uint8)
        mask_indicator_1 = instances[1]['mask_indicator'].array
        assert np.allclose(expected_mask_indicator_1, mask_indicator_1)


if __name__ == '__main__':
    unittest.main()
