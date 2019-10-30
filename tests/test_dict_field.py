
import numpy as np

from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField, ListField, ArrayField, SpanField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter


from kb.dict_field import DictField
from kb.entity_linking import TokenCharactersIndexerTokenizer

import unittest
import torch


class TestDictField(unittest.TestCase):
    def setUp(self):
        super(TestDictField, self).setUp()

        entity_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())

        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("entity1", "entity")
        self.vocab.add_token_to_namespace("entity2", "entity")
        self.vocab.add_token_to_namespace("entity3", "entity")
        self.entity_indexer = {"entity": TokenCharactersIndexerTokenizer(
            "entity", character_tokenizer=entity_tokenizer)
        }

        tokens1 = "The sentence .".split()
        tokens_field = TextField(
            [Token(t) for t in tokens1],
            token_indexers={'tokens': SingleIdTokenIndexer()}
        )

        self.instance1_fields = {
            "candidate_entities": TextField(
                    [Token("entity1 entity2"), Token("entity_unk")],
                    token_indexers=self.entity_indexer),
            "candidate_entity_prior": ArrayField(np.array([[0.5, 0.5], [1.0, 0.0]])),
            "candidate_spans": ListField(
                    [SpanField(0, 0, tokens_field),
                     SpanField(1, 2, tokens_field)]
            )
        }

        tokens2 = "The sentence".split()
        tokens2_field = TextField(
            [Token(t) for t in tokens2], 
            token_indexers={'tokens': SingleIdTokenIndexer()}
        )

        self.instance2_fields = {
            "candidate_entities": TextField(
                    [Token("entity1")], 
                    token_indexers=self.entity_indexer),
            "candidate_entity_prior": ArrayField(np.array([[1.0]])),
            "candidate_spans": ListField(
                    [SpanField(1, 1, tokens2_field)],
            )
        }

    def test_get_padding_lengths(self):
        field = DictField(self.instance1_fields)
        field.index(self.vocab)
        lengths = field.get_padding_lengths()
        self.assertDictEqual(
            lengths,
                {'candidate_entities*entity_length': 2,
                  'candidate_entities*num_token_characters': 2,
                   'candidate_entities*num_tokens': 2,
                   'candidate_entity_prior*dimension_0': 2,
                   'candidate_entity_prior*dimension_1': 2,
                   'candidate_spans*num_fields': 2}
        )

    def test_dict_field_can_handle_empty(self):
        field = DictField(self.instance1_fields)
        empty = field.empty_field()
        self.assertTrue(True)

    def _check_tensors(self, tensor, expected):
        self.assertListEqual(
            sorted(list(tensor.keys())), sorted(list(expected.keys()))
        )
        for key in tensor.keys():
            if key == 'candidate_entities':
                a = tensor[key]['entity']
                b = expected[key]['entity']
            else:
                a = tensor[key]
                b = expected[key]
            self.assertTrue(np.allclose(a.numpy(), b.numpy()))


    def test_dict_field_as_tensor(self):
        field = DictField(self.instance1_fields)
        field.index(self.vocab)
        tensor = field.as_tensor(field.get_padding_lengths())

        expected = {'candidate_entities': {'entity': torch.tensor([[2, 3],
        [1, 0]])}, 'candidate_entity_prior': torch.tensor([[0.5000, 0.5000],
        [1.0000, 0.0000]]), 'candidate_spans': torch.tensor([[0, 0],
        [1, 2]])}

        self._check_tensors(tensor, expected)

    def test_dict_field_can_iterator(self):
        from allennlp.data import Instance
        from allennlp.data.iterators import BasicIterator

        iterator = BasicIterator()
        iterator.index_with(self.vocab)

        instances = [
            Instance({"candidates": DictField(self.instance1_fields)}),
            Instance({"candidates": DictField(self.instance2_fields)})
        ]

        for batch in iterator(instances, num_epochs=1, shuffle=False):
            break

        expected_batch = {'candidates': {
            'candidate_entities': {'entity': torch.tensor([[[2, 3],
                                                      [1, 0]],

                                                     [[2, 0],
                                                      [0, 0]]])},
            'candidate_entity_prior': torch.tensor([[[0.5000, 0.5000],
                                               [1.0000, 0.0000]],

                                              [[1.0000, 0.0000],
                                               [0.0000, 0.0000]]]),
             'candidate_spans': torch.tensor([[[ 0,  0],
                                               [ 1,  2]],
            
                                              [[ 1,  1],
                                               [-1, -1]]])}
        }

        self._check_tensors(batch['candidates'], expected_batch['candidates'])

    def test_list_field_of_dict_field(self):
        from allennlp.data import Instance
        from allennlp.data.iterators import BasicIterator

        tokens3 = "The long sentence .".split()
        tokens3_field = TextField(
            [Token(t) for t in tokens3],
            token_indexers={'tokens': SingleIdTokenIndexer()}
        )

        instance3_fields = {
            "candidate_entities": TextField(
                    [Token("entity1 entity2 entity3"), Token("entity_unk"), Token("entity2 entity3")],
                    token_indexers=self.entity_indexer),
            "candidate_entity_prior": ArrayField(np.array([[0.1, 0.1, 0.8],
                                                           [1.0, 0.0, 0.0],
                                                           [0.33, 0.67, 0.0]])),
            "candidate_spans": ListField(
                    [SpanField(1, 1, tokens3_field), SpanField(1, 2, tokens3_field), SpanField(1, 3, tokens3_field)],
            )
        }

        iterator = BasicIterator()
        iterator.index_with(self.vocab)

        instances = [Instance({"candidates": ListField([
                                    DictField(self.instance1_fields),
                                    DictField(self.instance2_fields)])}),
                     Instance({"candidates": ListField([
                                    DictField(self.instance1_fields),
                                    DictField(instance3_fields)])})
        ]

        for batch in iterator(instances, num_epochs=1, shuffle=False):
            pass

        self.assertTrue(batch['candidates']['candidate_entities']['entity'].shape == batch['candidates']['candidate_entity_prior'].shape)


if __name__ == '__main__':
    unittest.main()

