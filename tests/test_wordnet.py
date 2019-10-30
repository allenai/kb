
import unittest
import numpy as np

import torch

from kb.wordnet import WordNetFineGrainedSenseDisambiguationReader
from kb.wordnet import WordNetSpacyPreprocessor, \
                                   WordNetCandidateMentionGenerator
from kb.wordnet import unpack_wsd_training_instance
from kb.wordnet import WordNetAllEmbedding


from kb.testing import get_wsd_fixture_batch, get_wsd_reader


class TestRemapSpanWordPiece(unittest.TestCase):
    def test_remap_span(self):
        reader, vocab, iterator = get_wsd_reader(True, use_bert_indexer=True)

        wsd_file = 'tests/fixtures/wordnet/wsd_dataset.json'
        instances = reader.read(wsd_file)
        for batch in iterator(instances, shuffle=False, num_epochs=1):
            break

        expected_spans = torch.tensor(
            [[[4, 5]],
             [[2, 2]]]
        )

        self.assertEqual(
            batch['candidate_spans'].numpy().tolist(),
            expected_spans.numpy().tolist()
        )


class TestWordNetFineGrainedSenseDisambiguationReader(unittest.TestCase):
    def _get_eval_batch(self):
        return {
            'tokens': {'tokens': torch.tensor([[2, 3, 4, 5, 6],
                                         [7, 5, 0, 0, 0]])},
            'candidate_entities': {'ids': torch.tensor([[[ 1,  0],
                                                   [23, 24]],
  
                                                  [[26, 25],
                                                   [ 0,  0]]])},
           'gold_entities': {'ids': torch.tensor([[[22],
                                             [23]],
                                          
                                             [[25],
                                              [ 0]]])},
           'gold_data_ids': [['d000.s000.t000', 'd000.s000.t001'], ['d000.s001.t000']],
           'candidate_spans': torch.tensor([[[ 0,  0],
                                       [ 3,  4]],

                                      [[ 1,  1],
                                       [-1, -1]]]),
            'candidate_entity_prior': torch.tensor(
                                       [[[1.0000, 0.0000],
                                         [0.2500, 0.7500]],

                                        [[0.4000, 0.6000],
                                         [0.0000, 0.0000]]]),
            'candidate_segment_ids': torch.tensor([[0, 0], [0, 0]]),
    }

    def _get_train_batch(self):
        return {
            'tokens': {'tokens': torch.tensor([[2, 3, 4, 5, 6],
                                         [7, 5, 0, 0, 0]])},
            'candidate_entities': {'ids': torch.tensor([[[23, 24,  0,  0]],
                                                   [[26, 25, 27, 28]]])},
            'gold_entities': {'ids': torch.tensor([[[23]],
                                             [[25]]])},
            'gold_data_ids': [['d000.s000.t001'], ['d000.s001.t000']],
            'candidate_spans': torch.tensor([[[3, 4]],
                                        [[1, 1]]]),
            'candidate_entity_prior': torch.tensor(
                        [[[0.2500, 0.7500, 0.0000, 0.0000]],
                         [[0.1429, 0.2143, 0.2857, 0.3571]]]),
            'candidate_segment_ids': torch.tensor([[0], [0]]),
        }

    def test_wordnet_fine_grained_reader_evaluation(self):
        batch = get_wsd_fixture_batch(False)
        expected_batch = self._get_eval_batch()
        self._check_batch(batch, expected_batch)

    def test_wordnet_fine_grained_reader_training(self):
        batch = get_wsd_fixture_batch(True)
        expected_batch = self._get_train_batch()
        self._check_batch(batch, expected_batch, exclude_candidate_entities=True)

    def _check_batch(self, batch, expected_batch, exclude_candidate_entities=False):
        self.assertEqual(
                sorted(list(batch.keys())),
                sorted(list(expected_batch.keys()))
        )
        self.assertEqual(
                batch['tokens']['tokens'].numpy().tolist(),
                expected_batch['tokens']['tokens'].numpy().tolist()
        )

        if not exclude_candidate_entities:
            # exclude candidate for training (so don't check the neg. samples)
            self.assertEqual(
                batch['candidate_entities']['ids'].numpy().tolist(),
                expected_batch['candidate_entities']['ids'].numpy().tolist()
            )

        self.assertEqual(
                batch['gold_entities']['ids'].numpy().tolist(),
                expected_batch['gold_entities']['ids'].numpy().tolist()
        )
        self.assertEqual(
                batch['candidate_spans'].numpy().tolist(),
                expected_batch['candidate_spans'].numpy().tolist()
        )

        self.assertTrue(
                np.allclose(batch['candidate_entity_prior'].numpy(),
                             expected_batch['candidate_entity_prior'].numpy(),
                atol=1e-4),
        )


class TestWordNetSpacyPreprocessor(unittest.TestCase):
    def test_wordnet_process(self):
        processor = WordNetSpacyPreprocessor()
        tokens = processor("Bob walked to the stores.")

        expected_text = "Bob walked to the stores .".split()
        expected_lemmas = "Bob walk to the store .".split()
        for k, token in enumerate(tokens):
            self.assertEqual(token.text, expected_text[k])
            self.assertEqual(token.lemma_, expected_lemmas[k])


class TestWordNetCandidateMentionGenerator(unittest.TestCase):
    def test_mention_generator(self):
        entity_file = 'tests/fixtures/wordnet/entities_fixture.jsonl'
        generator = WordNetCandidateMentionGenerator(entity_file)

        sentence = "Big cats are cats."

        candidates = generator.get_mentions_raw_text(sentence)

        expected_candidates = {
             'tokenized_text': ['Big', 'cats', 'are', 'cats', '.'],
             'candidate_spans': [[0, 1], [1, 1], [3, 3]],
             'candidate_entities': [['cat.n.01', 'cat.n.04'],
              ['computerized_tomography.n.01'],
              ['computerized_tomography.n.01']],
             'candidate_entity_priors': [[0.16666666666666666, 0.8333333333333334],
              [1.0],
              [1.0]]
        }
    
        self._check(candidates, expected_candidates)

    def test_mention_generator_no_candidates(self):
        entity_file = 'tests/fixtures/wordnet/entities_fixture.jsonl'
        generator = WordNetCandidateMentionGenerator(entity_file)
        candidates = generator.get_mentions_raw_text(".")
        expected_candidates = {'tokenized_text': ['.'],
                              'candidate_spans': [[-1, -1]],
                              'candidate_entities': [["@@PADDING@@"]],
                              'candidate_entity_priors': [[1.0]]}
        self._check(candidates, expected_candidates)

    def _check(self, candidates, expected_candidates):
        self.assertEqual(sorted(expected_candidates.keys()),
                         sorted(candidates.keys()))

        for key in expected_candidates:
            if key != 'candidate_entity_priors':
                self.assertEqual(candidates[key], expected_candidates[key])
            else:
                self.assertEqual(len(expected_candidates[key]), len(candidates[key]))
                for i in range(len(expected_candidates[key])):
                    self.assertEqual(len(expected_candidates[key][i]), len(candidates[key][i]))
                    for j in range(len(expected_candidates[key][i])):
                        self.assertAlmostEqual(candidates[key][i][j],
                                           expected_candidates[key][i][j])


    def _setup_stuff(self):
        entity_file = 'tests/fixtures/wordnet/entities_fixture.jsonl'
        generator = WordNetCandidateMentionGenerator(entity_file, use_surface_form=True)

        gold_annotations = {
            'tokenized_text': ['Mr.', 'A', 'saw', 'a', 'person', 'with', 'half', '-', 'baked', 'hot', 'dogs'],
            'gold_spans': [
                [0, 1],
                [2, 3],
                [4, 4],
                [6, 8],
                [9, 10]
            ],
            'gold_lemma_ids': [
                'person%1:03:00::',    # PER synset
                'see_a%2:05:25::',
                'person%1:01:55::',
                'half-baked%3:01:22::',
                'hot_dog%1:01:06::',
            ],
            'gold_lemmas': [
                'person',
                'see_a',
                'person',
                'half-baked',
                'hot_dog'
            ],
            'gold_pos': [
                'NOUN',
                'VERB',
                'NOUN',
                'ADJ',
                'NOUN',
            ],
        }

        return generator, gold_annotations

    def test_filter_with_gold_annotations_training(self):
        generator, gold_annotations = self._setup_stuff()
        candidates = generator.get_mentions_with_gold_spans(gold_annotations)

        expected_candidates = {
             'candidate_entities': [['see_a.v.01', 'see_a.a.02'],
                                    ['person.n.01', 'person.n.02', 'person.n.03'],
                                    ['half-baked.a.01'],
                                    ['hot_dog.n.01']],
             'candidate_entity_priors': [[0.5, 0.5],
                                         [0.23076923076923078,
                                          0.34615384615384615,
                                          0.4230769230769231],
                                         [1.0],
                                         [1.0]],
             'candidate_spans': [[2, 3], [4, 4], [6, 8], [9, 10]],
             'tokenized_text': ['Mr.',
                                'A',
                                'saw',
                                'a',
                                'person',
                                'with',
                                'half',
                                '-',
                                'baked',
                                'hot',
                                'dogs']}

        self._check(candidates, expected_candidates)

    def test_filter_with_gold_annotations_test(self):
        generator, gold_annotations = self._setup_stuff()
        candidates = generator.get_mentions_from_gold_span_lemma_pos(gold_annotations)

        expected_candidates = {
             'candidate_entities': [['person.n.01'],
                                    ['see_a.v.01'],
                                    ['person.n.01', 'person.n.02', 'person.n.03'],
                                    ['half-baked.a.01'],
                                    ['hot_dog.n.01']],
             'candidate_entity_priors': [[1.0],
                                         [1.0],
                                         [0.23076923076923078,
                                          0.34615384615384615,
                                          0.4230769230769231],
                                         [1.0],
                                         [1.0]],
             'candidate_spans': [[0, 1], [2, 3], [4, 4], [6, 8], [9, 10]],
             'tokenized_text': ['Mr.',
                                'A',
                                'saw',
                                'a',
                                'person',
                                'with',
                                'half',
                                '-',
                                'baked',
                                'hot',
                                'dogs']
        }

        self._check(candidates, expected_candidates)


class TestUnpackWsdInstance(unittest.TestCase):
    def test_unpack_wsd_training_instance(self):
        context = [
            {'token': 'It'},
            {'token': 'is'},
            {'token': 'half-baked', 'pos': 'NOUN', 'senses': ['1:00:01::'],
                      'lemma': 'half-baked', 'id': 'd000.s000.t000'},
            {'token': 'said', 'pos': 'VERB', 'senses': ['2:00:05::'],
                      'lemma': 'say', 'id': 'd000.s000.t001'},
            {'token': 'New York', 'pos': 'NOUN', 'senses': ['1:04:00::'],
                      'lemma': 'new_york', 'id': 'd000.s000.t002'},
            {'token': '.'}
        ]

        unpacked = unpack_wsd_training_instance(context)

        expected_unpacked = {
                 'tokenized_text': ['It',
                  'is',
                  'half',
                  '-',
                  'baked',
                  'said',
                  'New',
                  'York',
                  '.'],
                 'gold_spans': [[2, 4], [5, 5], [6, 7]],
                 'gold_lemma_ids': ['half-baked%1:00:01::',
                  'say%2:00:05::',
                  'new_york%1:04:00::'],
                 'gold_lemmas': ['half-baked', 'say', 'new_york'],
                 'gold_pos': ['NOUN', 'VERB', 'NOUN'],
                 'gold_ids': ['d000.s000.t000', 'd000.s000.t001', 'd000.s000.t002']}

        self.assertEqual(unpacked, expected_unpacked)


class TestWordNetAllEmbedding(unittest.TestCase):
    def test_wordnet_all_embedding(self):
        entity_file = 'tests/fixtures/wordnet/entities_cat_hat.jsonl'
        vocab_file = 'tests/fixtures/wordnet/cat_hat_synset_mask_null_vocab.txt'
        embedding_file = 'tests/fixtures/wordnet/cat_hat_mask_null_embedding.hdf5'

        all_embed = WordNetAllEmbedding(
                 entity_file=entity_file,
                 embedding_file=embedding_file,
                 vocab_file=vocab_file,
                 entity_dim=18,
                 pos_embedding_dim=5)

        entity_ids = torch.tensor([[[ 5,  0],
                                    [7, 11]],

                                    [[3, 4],
                                    [ 0,  0]]])

        embeds = all_embed(entity_ids)
        self.assertEqual(list(embeds.shape), [2, 2, 2, 18])

    def test_wordnet_all_embedding_no_pos(self):
        embedding_file = 'tests/fixtures/wordnet/cat_hat_mask_null_embedding.hdf5'

        all_embed = WordNetAllEmbedding(
                 embedding_file=embedding_file,
                 entity_dim=18,
                 pos_embedding_dim=None)

        entity_ids = torch.tensor([[[ 5,  0],
                                    [7, 11]],

                                    [[3, 4],
                                    [ 0,  0]]])

        embeds = all_embed(entity_ids)
        self.assertEqual(list(embeds.shape), [2, 2, 2, 18])



if __name__ == '__main__':
    unittest.main()
