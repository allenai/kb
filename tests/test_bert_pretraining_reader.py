
from kb.bert_pretraining_reader import BertPreTrainingReader, \
    replace_candidates_with_mask_entity
from kb.wordnet import WordNetCandidateMentionGenerator
from kb.wiki_linking_util import WikiCandidateMentionGenerator

from kb.testing import get_bert_pretraining_reader_with_kg as get_reader

from kb.bert_tokenizer_and_candidate_generator import BertTokenizerAndCandidateGenerator

import unittest

from allennlp.common import Params
from allennlp.data import DatasetReader, Vocabulary, DataIterator

import numpy as np

import torch



class TestReplaceCandidatesWithMaskEntity(unittest.TestCase):
    def test_replace_candidates_with_mask_entity(self):
        spans_to_mask = set([(0, 0), (1, 2), (3, 3)])
        candidates = {
            'wordnet': {'candidate_spans': [[0, 0], [1, 1], [1, 2]],
                        'candidate_entities': [["a"], ["b", "c"], ["d"]],
                        'candidate_entity_priors': [[1.0], [0.2, 0.8], [1.0]]},
            'wiki': {'candidate_spans': [[3, 3]],
                     'candidate_entities': [["d"]],
                     'candidate_entity_priors': [[1.0]]},
        }
        replace_candidates_with_mask_entity(candidates, spans_to_mask)

        expected_candidates = {
            'wordnet': {'candidate_spans': [[0, 0], [1, 1], [1, 2]],
                        'candidate_entities': [["@@MASK@@"], ["b", "c"], ["@@MASK@@"]],
                        'candidate_entity_priors': [[1.0], [0.2, 0.8], [1.0]]},
            'wiki': {'candidate_spans': [[3, 3]],
                     'candidate_entities': [["@@MASK@@"]],
                     'candidate_entity_priors': [[1.0]]},
        }

        for key in ['wordnet', 'wiki']:
            for key2 in ['candidate_spans', 'candidate_entities']:
                self.assertListEqual(
                    candidates[key][key2], expected_candidates[key][key2]
                )


class TestBertPretrainingReader(unittest.TestCase):
    def test_create_masked_lm_predictions(self):
        reader = get_reader(masked_lm_prob=0.5)
        np.random.seed(5)

        tokens, lm_labels = reader._tokenizer_masker.create_masked_lm_predictions(
                "The original tokens in the sentence .".split()
        )

        expected_tokens = ['The', '[MASK]', '[MASK]', 'in', '[MASK]', 'sentence', '[MASK]']
        expected_lm_labels = ['[PAD]', 'original', 'tokens', '[PAD]', 'the', '[PAD]', '.']

        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_lm_labels, lm_labels)

    def test_reader_can_run_with_full_mask_strategy(self):
        reader = get_reader('full_mask', masked_lm_prob=0.5)
        instances = reader.read("tests/fixtures/bert_pretraining/shard1.txt")
        self.assertEqual(len(instances), 2)

    def test_reader_can_run_with_wordnet_and_wiki(self):
        reader = get_reader('full_mask', masked_lm_prob=0.5, include_wiki=True)
        instances = reader.read("tests/fixtures/bert_pretraining/shard1.txt")
        self.assertEqual(len(instances), 2)

    def test_reader(self):
        reader = get_reader(masked_lm_prob=0.15)

        np.random.seed(5)
        instances = reader.read("tests/fixtures/bert_pretraining/shard1.txt")

        vocab = Vocabulary.from_params(Params({
            "directory_path": "tests/fixtures/bert/vocab_dir_with_entities_for_tokenizer_and_generator"
        }))
        iterator = DataIterator.from_params(Params({"type": "basic"}))
        iterator.index_with(vocab)

        for batch in iterator(instances, num_epochs=1, shuffle=False):
            break

        actual_tokens_ids = batch['tokens']['tokens']
        expected_tokens_ids = torch.tensor(
                [[16, 18, 19, 20,  1, 19, 21, 13, 17, 21,  3,  4, 12, 13, 17],
                [16,  1, 13, 17, 21,  1,  1, 13, 17,  0,  0,  0,  0,  0,  0]])

        self.assertEqual(actual_tokens_ids.tolist(), expected_tokens_ids.tolist())

        actual_entities = batch['candidates']['wordnet']['candidate_entities']['ids']
        expected_entities = torch.tensor(
                    [[[29, 30],
                     [31,  0],
                     [31,  0]],
                   
                    [[ 0,  0],
                     [ 0,  0],
                     [ 0,  0]]])
        self.assertEqual(actual_entities.tolist(), expected_entities.tolist())

        expected_spans = torch.tensor(
                       [[[ 1,  3],
                         [ 2,  3],
                         [ 5,  6]],
                
                        [[-1, -1],
                         [-1, -1],
                         [-1, -1]]])
        actual_spans = batch['candidates']['wordnet']['candidate_spans']
        self.assertEqual(actual_spans.tolist(), expected_spans.tolist())

        expected_lm_labels = torch.tensor(
                [[ 0,  0,  0,  0,  0,  0, 20,  0,  0,  2,  0,  0,  0,  0,  0],
                 [ 0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
        actual_lm_labels = batch['lm_label_ids']['lm_labels']
        self.assertEqual(actual_lm_labels.tolist(), expected_lm_labels.tolist())

        expected_segment_ids = torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])
        self.assertEqual(batch['segment_ids'].tolist(), expected_segment_ids.tolist())
        self.assertTrue(batch['segment_ids'].dtype == torch.long)


if __name__ == '__main__':
    unittest.main()


