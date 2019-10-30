
import unittest
import torch

from kb.kg_embedding import KGTupleReader, get_labels_tensor_from_indices, \
        RankingAndHitsMetric

from allennlp.data import Vocabulary
from allennlp.common import Params
from allennlp.data.iterators import BasicIterator


class TestRankingAndHitsMetric(unittest.TestCase):
    def test_ranking_and_hits(self):
        batch_size = 2
        num_entities = 5

        predicted = torch.rand(batch_size, num_entities)
        all_entity2 = torch.LongTensor([[2, 3, 4], [1, 0, 0]])
        entity2 = torch.LongTensor([3, 2])

        metric = RankingAndHitsMetric()
        metric(predicted, all_entity2, entity2)
        metrics = metric.get_metric()

        self.assertTrue(True)


class TestLabelFromIndices(unittest.TestCase):
    def test_get_labels_tensor_from_indices(self):
        batch_size = 2
        num_embeddings = 7
        entity_ids = torch.tensor([[5, 6, 1, 0, 0, 0], [1, 2, 3, 4, 5, 6]])
        labels = get_labels_tensor_from_indices(
                batch_size, num_embeddings, entity_ids
        )

        expected_labels = torch.tensor(
                [[0., 1., 0., 0., 0., 1., 1.],
                 [0., 1., 1., 1., 1., 1., 1.]]
        )

        self.assertTrue(torch.abs(labels - expected_labels).max() < 1e-6)

    def test_get_labels_tensor_from_indices_with_smoothing(self):
        batch_size = 2
        num_embeddings = 7
        entity_ids = torch.tensor([[5, 6, 1, 0, 0, 0], [1, 2, 3, 4, 5, 6]])
        labels = get_labels_tensor_from_indices(
                batch_size, num_embeddings, entity_ids, label_smoothing=0.1
        )

        expected_labels = torch.tensor(
                [[0., 1., 0., 0., 0., 1., 1.],
                 [0., 1., 1., 1., 1., 1., 1.]]
        )
        smoothed_labels = (1.0 - 0.1) * expected_labels + 0.1 / 7 * torch.tensor([[3.0], [6.0]])

        self.assertTrue(torch.abs(labels - smoothed_labels).max() < 1e-6)


class TestKGTupleReader(unittest.TestCase):
    def _check_batch(self, batch, vocab, expected_entity, expected_relation, expected_entity2):
        expected_entity_ids = [vocab.get_token_index(str(e), 'entity')
                               for e in expected_entity]
        self.assertListEqual(batch['entity']['entity'].flatten().tolist(),
                             expected_entity_ids)

        expected_relation_ids = [vocab.get_token_index(r, 'relation')
                               for r in expected_relation]
        self.assertListEqual(batch['relation']['relation'].flatten().tolist(),
                             expected_relation_ids)

        # check the entity2
        expected_entity2_ids = [
            [vocab.get_token_index(str(e), 'entity') for e in ee]
            for ee in expected_entity2
        ]
        self.assertEqual(len(expected_entity2), batch['entity2']['entity'].shape[0])
        for k in range(len(expected_entity2)):
            self.assertListEqual(
                sorted(expected_entity2_ids[k]),
                sorted([e for e in batch['entity2']['entity'][k].tolist() if e != 0])
            )

    def test_no_eval(self):
        reader = KGTupleReader()
        instances = reader.read('tests/fixtures/kg_embeddings/wn18rr_train.txt')

        self.assertTrue(len(instances) == 8)

        # create the vocab and index to make sure things look good
        vocab = Vocabulary.from_params(Params({}), instances)
        # (+2 for @@PADDING@@ and @@UNKNOWN@@
        self.assertEqual(vocab.get_vocab_size("entity"), 5 + 2)
        self.assertEqual(vocab.get_vocab_size("relation"), 4 + 2)

        # now get a batch
        iterator = BasicIterator(batch_size=32)
        iterator.index_with(vocab)
        for batch in iterator(instances, num_epochs=1, shuffle=False):
            pass

        # check it!
        expected_entity = [1, 2, 1, 3, 3, 4, 1, 5]
        expected_relation = ['_hypernym', '_hypernym_reverse',
                              '_derivationally_related_form', '_derivationally_related_form_reverse',
                              '_hypernym_reverse', '_hypernym', '_hypernym_reverse',
                              '_hypernym_reverse']
        expected_entity2 = [[2, 3], [1], [3], [1], [1], [1, 5], [4], [4]]

        self._check_batch(batch, vocab,
                          expected_entity, expected_relation, expected_entity2)

    def test_kg_reader_with_eval(self):
        train_file = 'tests/fixtures/kg_embeddings/wn18rr_train.txt'
        dev_file = 'tests/fixtures/kg_embeddings/wn18rr_dev.txt'

        train_instances = KGTupleReader().read(train_file)

        reader = KGTupleReader(extra_files_for_gold_pairs=[train_file])
        instances = reader.read(dev_file)
        self.assertEqual(len(instances), 2)

        vocab = Vocabulary.from_params(Params({}), train_instances + instances)
        iterator = BasicIterator(batch_size=32)
        iterator.index_with(vocab)
        for batch in iterator(instances, num_epochs=1, shuffle=False):
            pass

        expected_entity = [1, 5]
        expected_relation = ['_hypernym', '_hypernym_reverse']
        expected_entity2 = [[5, 2, 3], [1, 4]]
        self._check_batch(batch, vocab,
                          expected_entity, expected_relation, expected_entity2)


if __name__ == '__main__':
    unittest.main()

