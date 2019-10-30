
import unittest

import torch

from kb.self_attn_bucket_iterator import SelfAttnBucketIterator
from allennlp.data.fields import TextField
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer


class TestSelfAttnBucketIterator(unittest.TestCase):
    def test_self_attn_iterator(self):
        indexer = {'tokens': SingleIdTokenIndexer()}

        # make some instances
        instances = []
        for k in range(100):
            l = max(int(torch.rand(1).item() * 500), 1)
            instances.append(Instance(
                {'tokens': TextField(
                    [Token('a') for i in range(l)], token_indexers=indexer)})
            )

        schedule = [[16, 128], [8, 256], [4, 512]]

        sub_iterator = BucketIterator(
                batch_size=16,
                sorting_keys=[['tokens', 'num_tokens']],
                padding_noise=0.0
        )

        it = SelfAttnBucketIterator(schedule, sub_iterator)
        it.index_with(Vocabulary())

        batches = [batch for batch in it(instances, num_epochs=1)]

        n_instances = 0
        for batch in batches:
            batch_size = batch['tokens']['tokens'].shape[0]
            n_instances += batch_size
            timesteps = batch['tokens']['tokens'].shape[1]
            if timesteps <= 128:
                expected_batch_size = 16
            elif timesteps <= 256:
                expected_batch_size = 8
            else:
                expected_batch_size = 4
            # batch might be smaller then expected if we split a larger batch
            # and the sequence length for the shorter segment falls into a lower
            # bucket
            self.assertTrue(batch_size <= expected_batch_size)

        self.assertEqual(n_instances, 100)

if __name__ == '__main__':
    unittest.main()

