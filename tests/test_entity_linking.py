
import unittest

import torch
import numpy as np

from allennlp.data import TokenIndexer, Vocabulary, Token
from allennlp.common import Params
from allennlp.models import Model

from kb.entity_linking import TokenCharactersIndexerTokenizer
from kb.entity_linking import remap_span_indices_after_subword_tokenization
from kb.testing import get_bert_test_fixture
from kb.include_all import ModelArchiveFromParams


class TestRemapAfterWordpiece(unittest.TestCase):
    def test_remap(self):
        bert_fixture = get_bert_test_fixture()
        indexer = bert_fixture['indexer']

        tokens = [Token(t) for t in 'The words dog overst .'.split()]
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(tokens, vocab, 'wordpiece')

        original_span_indices = [
            [0, 0], [0, 1], [2, 3], [3, 3], [2, 4]
        ]
        offsets = indexed['wordpiece-offsets']

        expected_remapped = [
            [1, 1],
            [1, 2],
            [3, 5],
            [4, 5],
            [3, 6]
        ]

        remapped = remap_span_indices_after_subword_tokenization(
                original_span_indices, offsets, len(indexed['wordpiece'])
        )

        self.assertEqual(expected_remapped, remapped)


class TestTokenCharactersIndexerTokenizer(unittest.TestCase):

    def test_token_characters_indexer_tokenizer(self):
        params = Params({
            "type": "characters_tokenizer",
            "tokenizer": {
                "type": "word",
                "word_splitter": {"type": "just_spaces"},
            },
            "namespace": "tok"
        })

        indexer = TokenIndexer.from_params(params)

        vocab = Vocabulary()
        vocab.add_token_to_namespace("the", namespace="tok")
        vocab.add_token_to_namespace("2", namespace="tok")

        indices = indexer.tokens_to_indices(
                [Token(t) for t in "the 2 .".split()], vocab, 'a'
        )

        self.assertListEqual(indices['a'], [[2], [3], [1]])



if __name__ == '__main__':
    unittest.main()

