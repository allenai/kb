import unittest
import torch

from kb.bert_tokenizer_and_candidate_generator import BertTokenizerAndCandidateGenerator
from kb.wordnet import WordNetCandidateMentionGenerator

from allennlp.common import Params
from allennlp.data import TokenIndexer, Vocabulary, DataIterator, Instance


def _get_wordnet_generator():
    return WordNetCandidateMentionGenerator(
            'tests/fixtures/wordnet/entities_fixture.jsonl')


def _get_entity_indexers():
    indexer_params = Params({
            "type": "characters_tokenizer",
            "tokenizer": {
                    "type": "word",
                    "word_splitter": {"type": "just_spaces"},
            },
            "namespace": "entity"
    })
    return {'wordnet': TokenIndexer.from_params(indexer_params)}


def get_bert_tokenizer_and_candidate_generator(whitespace_tokenize=True, max_word_piece_sequence_length=512):
    return BertTokenizerAndCandidateGenerator(
            {'wordnet': _get_wordnet_generator()},
            _get_entity_indexers(),
            bert_model_type="tests/fixtures/bert/vocab.txt",
            do_lower_case=True,
            max_word_piece_sequence_length=max_word_piece_sequence_length,
            whitespace_tokenize=whitespace_tokenize
    )


class TestBertTokenizerAndCandidateGenerator(unittest.TestCase):

    def test_with_no_candidate_generators(self):
        candidate_generator = BertTokenizerAndCandidateGenerator({}, {}, bert_model_type="tests/fixtures/bert/vocab.txt", do_lower_case=True)
        actual = candidate_generator.tokenize_and_generate_candidates("this is a sentence")

        assert actual == {'tokens': ['[CLS]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[SEP]'],
                          'segment_ids': [0, 0, 0, 0, 0, 0],
                          'candidates':{},
                          'offsets_a': [2, 3, 4, 5],
                          'offsets_b': None}
        _ = candidate_generator.convert_tokens_candidates_to_fields(actual)

    def test_candidate_tokenize(self):
        candidate_generator = get_bert_tokenizer_and_candidate_generator(whitespace_tokenize=True)
        actual = candidate_generator._tokenize_text(text='Big cats are cats .')
        # [offsets, grouped_wp_tokens, tokens]
        expected = [
                [1, 3, 4, 6, 7],
                [['big'], ['cat', '##s'], ['[UNK]'], ['cat', '##s'], ['.']],
                ['Big', 'cats', 'are', 'cats', '.']
        ]
        self.assertEqual(list(actual), expected)

        candidate_generator = get_bert_tokenizer_and_candidate_generator(whitespace_tokenize=False)
        actual = candidate_generator._tokenize_text(text='Big cats are cats.')
        # [offsets, grouped_wp_tokens, tokens]
        self.assertEqual(list(actual), expected)

    def test_generate_sentence_entity_candidates(self):
        candidate_generator = get_bert_tokenizer_and_candidate_generator(whitespace_tokenize=True)

        offsets, grouped_wp_tokens, tokens = \
            candidate_generator._tokenize_text(text='Big cats are cats .')
        results = candidate_generator._generate_sentence_entity_candidates(
                tokens=tokens,
                offsets=offsets)

        expected_candidate_entities = [
                ['cat.n.01', 'cat.n.04'],
                ['computerized_tomography.n.01'],
                ['computerized_tomography.n.01']
        ]
        expected_spans = [[0, 2], [1, 2], [4, 5]]

        self.assertEqual(expected_spans, results['wordnet']['candidate_spans'])
        self.assertEqual(expected_candidate_entities, results['wordnet']['candidate_entities'])

    def test_tokenize_and_generate_sentence_entity_candidates(self):
        candidate_generator = get_bert_tokenizer_and_candidate_generator(max_word_piece_sequence_length=15,
                                                                         whitespace_tokenize=False)

        text_a = 'Big cats are cats.'
        text_b = 'Bigger cats are even quick than cats are are are are are are are are.'
        fields = candidate_generator.tokenize_and_generate_candidates(text_a=text_a, text_b=text_b)

        expected_wp_tokens = ['[CLS]', 'big', 'cat', '##s', '[UNK]', 'cat', '##s', '[SEP]', '[UNK]', 'cat',
                              '##s', '[UNK]', '[UNK]', 'quick', '[SEP]']
        expected_segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        expected_offsets_a = [2, 4, 5, 7]
        expected_offsets_b = [9, 11, 12, 13, 14]

        self.assertEqual(expected_wp_tokens, fields['tokens'])
        self.assertEqual(expected_segment_ids, fields['segment_ids'])
        self.assertEqual(expected_offsets_a, fields['offsets_a'])
        self.assertEqual(expected_offsets_b, fields['offsets_b'])

        expected_strings = ['bigcat##s', 'cat##s', 'cat##s', 'cat##s']
        candidate_spans = fields['candidates']['wordnet']['candidate_spans']
        self.assertEqual(len(expected_strings), len(candidate_spans))
        for string, (start, end) in zip(expected_strings, candidate_spans):
            self.assertEqual(string, ''.join(expected_wp_tokens[start: end + 1]))

        expected_candidates = [['cat.n.01', 'cat.n.04'],
                               ['computerized_tomography.n.01'],
                               ['computerized_tomography.n.01'],
                               ['computerized_tomography.n.01']]
        self.assertEqual(expected_candidates, fields['candidates']['wordnet']['candidate_entities'])

        expected_span_segment_ids = [0, 0, 0, 1]
        self.assertEqual(expected_span_segment_ids, fields['candidates']['wordnet']['candidate_segment_ids'])

        candidate_generator = get_bert_tokenizer_and_candidate_generator(max_word_piece_sequence_length=13,
                                                                         whitespace_tokenize=False)
        text_a = 'Big cats are cats.'
        text_b = 'Bigger cats are even quick than cats.'
        fields = candidate_generator.tokenize_and_generate_candidates(text_a=text_a, text_b=text_b)

        expected_wp_tokens = ['[CLS]', 'big', 'cat', '##s', '[UNK]', '[SEP]', '[UNK]', 'cat', '##s', '[UNK]',
                              '[UNK]', 'quick', '[SEP]']
        expected_segment_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        expected_offsets_a = [2, 4, 5]
        expected_offsets_b = [7, 9, 10, 11, 12]

        self.assertEqual(expected_wp_tokens, fields['tokens'])
        self.assertEqual(expected_segment_ids, fields['segment_ids'])
        self.assertEqual(expected_offsets_a, fields['offsets_a'])
        self.assertEqual(expected_offsets_b, fields['offsets_b'])

        expected_strings = ['bigcat##s', 'cat##s', 'cat##s']
        candidate_spans = fields['candidates']['wordnet']['candidate_spans']
        self.assertEqual(len(expected_strings), len(candidate_spans))
        for string, (start, end) in zip(expected_strings, candidate_spans):
            self.assertEqual(string, ''.join(expected_wp_tokens[start: end + 1]))

        expected_candidates = [['cat.n.01', 'cat.n.04'],
                               ['computerized_tomography.n.01'],
                               ['computerized_tomography.n.01']]
        self.assertEqual(expected_candidates, fields['candidates']['wordnet']['candidate_entities'])
        expected_span_segment_ids = [0, 0, 1]
        self.assertEqual(expected_span_segment_ids, fields['candidates']['wordnet']['candidate_segment_ids'])

        candidate_generator = get_bert_tokenizer_and_candidate_generator(max_word_piece_sequence_length=11,
                                                                         whitespace_tokenize=False)

        text_a = 'Bigger cats are even quick than cats are are are are are are are are.'
        fields = candidate_generator.tokenize_and_generate_candidates(text_a=text_a)

        expected_wp_tokens = ['[CLS]', '[UNK]', 'cat', '##s', '[UNK]', '[UNK]', 'quick', '[UNK]', 'cat', '##s', '[SEP]']
        expected_segment_ids = [0 for _ in expected_wp_tokens]
        expected_offsets_a = [2, 4, 5, 6, 7, 8, 10]

        self.assertEqual(expected_wp_tokens, fields['tokens'])
        self.assertEqual(expected_segment_ids, fields['segment_ids'])
        self.assertEqual(expected_offsets_a, fields['offsets_a'])
        self.assertIsNone(fields['offsets_b'])

        expected_strings = ['cat##s', 'cat##s']
        candidate_spans = fields['candidates']['wordnet']['candidate_spans']
        self.assertEqual(len(expected_strings), len(candidate_spans))
        for string, (start, end) in zip(expected_strings, candidate_spans):
            self.assertEqual(string, ''.join(expected_wp_tokens[start: end + 1]))

        expected_candidates = [['computerized_tomography.n.01'],
                               ['computerized_tomography.n.01']]
        self.assertEqual(expected_candidates, fields['candidates']['wordnet']['candidate_entities'])

    def test_null_entities(self):
        bert_t_and_cg = get_bert_tokenizer_and_candidate_generator()
        candidates = bert_t_and_cg.tokenize_and_generate_candidates("the .")
        self.assertEqual(candidates['candidates']['wordnet']['candidate_entities'], [['@@PADDING@@']])
        self.assertEqual(candidates['candidates']['wordnet']['candidate_spans'], [[-1, -1]])

    def test_null_entities_a_and_b(self):
        bert_t_and_cg = get_bert_tokenizer_and_candidate_generator()
        candidates = bert_t_and_cg.tokenize_and_generate_candidates("the .", "the .")
        self.assertEqual(candidates['candidates']['wordnet']['candidate_entities'], [['@@PADDING@@']])
        self.assertEqual(candidates['candidates']['wordnet']['candidate_spans'], [[-1, -1]])

    def test_null_entities_a_and_b(self):
        bert_t_and_cg = get_bert_tokenizer_and_candidate_generator()
        candidates = bert_t_and_cg.tokenize_and_generate_candidates("cat", "the .")
        self.assertEqual(
            candidates['candidates']['wordnet']['candidate_entities'],
            [['computerized_tomography.n.01']]
        )
        self.assertEqual(
            candidates['candidates']['wordnet']['candidate_spans'], [[1, 1]]
        )

    def test_convert_tokens_candidates_to_fields(self):
        # we'll create the fields, then make allennlp batch them into
        # tensors to test things end-to-end
        bert_t_and_cg = get_bert_tokenizer_and_candidate_generator()

        # two instances
        tokens_and_candidates = [
                {
                        'tokens': ['[CLS]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '.', '[SEP]'],
                        'segment_ids': [0] * 7,
                        'candidates': {
                                'wordnet': {
                                        'candidate_spans': [[1, 2], [2, 2], [4, 4]],
                                        'candidate_entities': [['cat.n.01', 'cat.n.04'],
                                                               ['computerized_tomography.n.01'],
                                                               ['computerized_tomography.n.01']],
                                        'candidate_entity_priors': [[0.16666666666666666, 0.8333333333333334],
                                                                    [1.0],
                                                                    [1.0]],
                                        'candidate_segment_ids': [0, 0, 0]
                                }
                        }
                },

                {
                        'tokens': ['[CLS]', 'quick', '##est', '[UNK]', '.', '[SEP]'],
                        'segment_ids': [0] * 6,
                        'candidates': {
                                'wordnet': {
                                        'candidate_spans': [[3, 3]],
                                        'candidate_entities': [['hat.n.01', 'hat.n.02', 'hat-trick.n.01']],
                                        'candidate_entity_priors': [[0.2, 0.3, 0.5]],
                                        'candidate_segment_ids': [0],
                                }
                        }
                }]

        instances = []
        for tokens_candidates in tokens_and_candidates:
            fields = bert_t_and_cg.convert_tokens_candidates_to_fields(
                    tokens_candidates
            )
            instances.append(Instance(fields))

        vocab = Vocabulary.from_params(Params({
                "directory_path": "tests/fixtures/bert/vocab_dir_with_entities_for_tokenizer_and_generator"
        }))
        iterator = DataIterator.from_params(Params({"type": "basic"}))
        iterator.index_with(vocab)

        for batch in iterator(instances, num_epochs=1, shuffle=False):
            break

        expected_batch = {'tokens': {
                'tokens': torch.tensor([[16,  1,  1,  1,  1, 13, 17],
                                        [16,  3,  4,  1, 13, 17,  0]])},
                'segment_ids': torch.tensor([[0,  0,  0,  0,  0, 0, 0],
                                             [0,  0,  0,  0,  0, 0, 0]]),
                'candidates': {'wordnet': {
                        'candidate_entity_priors': torch.tensor([[[0.1667, 0.8333, 0.0000],
                                                                  [1.0000, 0.0000, 0.0000],
                                                                  [1.0000, 0.0000, 0.0000]],

                                                                 [[0.2000, 0.3000, 0.5000],
                                                                  [0.0000, 0.0000, 0.0000],
                                                                  [0.0000, 0.0000, 0.0000]]]),
                        'candidate_entities': {'ids': torch.tensor([[[29, 30, 0],
                                                                     [31, 0, 0],
                                                                     [31, 0, 0]],

                                                                    [[25, 26, 24],
                                                                     [0, 0, 0],
                                                                     [0, 0, 0]]])},
                        'candidate_spans': torch.tensor([[[1, 2],
                                                          [2, 2],
                                                          [4, 4]],

                                                         [[3, 3],
                                                          [-1, -1],
                                                          [-1, -1]]]),
                        'candidate_segment_ids': torch.tensor([[0, 0, 0], [0, 0, 0]])}}}

        def __check(a, b):
            if a.dtype == torch.int64:
                self.assertTrue((a == b).all().item() == 1)
            else:
                self.assertTrue((torch.abs(a - b) < 1e-4).all().item() == 1)

        __check(expected_batch['tokens']['tokens'], batch['tokens']['tokens'])
        __check(expected_batch['segment_ids'], batch['segment_ids'])
        __check(expected_batch['candidates']['wordnet']['candidate_entity_priors'],
                batch['candidates']['wordnet']['candidate_entity_priors'])
        __check(expected_batch['candidates']['wordnet']['candidate_entities']['ids'],
                batch['candidates']['wordnet']['candidate_entities']['ids'])
        __check(expected_batch['candidates']['wordnet']['candidate_spans'],
                batch['candidates']['wordnet']['candidate_spans'])


if __name__ == '__main__':
    unittest.main()
