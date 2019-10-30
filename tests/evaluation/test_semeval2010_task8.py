
import unittest
from kb.include_all import SemEval2010Task8Reader, SemEval2010Task8Metric
from allennlp.common import Params
from allennlp.data import DatasetReader, DataIterator, Vocabulary
import torch


class TestSemEval2010Task8Metric(unittest.TestCase):
    def test_semeval2010_metric(self):
        predicted_ids = [
            torch.tensor([0, 15, 3]),
            torch.tensor([7, 0])
        ]
        gold_ids = [
            torch.tensor([0, 3, 3]),
            torch.tensor([0, 7])
        ]

        expected_f1 = 33.33

        metric = SemEval2010Task8Metric()
        for p, g in zip(predicted_ids, gold_ids):
            metric(p, g)

        f1 = metric.get_metric()

        self.assertAlmostEqual(expected_f1, f1)


class TestSemEval2010Task8Reader(unittest.TestCase):
    def test_semeval2010_task8_reader(self):
        reader_params = Params({
            "type": "semeval2010_task8",
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
        })

        reader = DatasetReader.from_params(reader_params)
        train_file = 'tests/fixtures/evaluation/semeval2010_task8/semeval2010_task8.json'

        instances = reader.read(train_file)

        # check that the offsets are right!
        segment_ids = instances[0]['segment_ids'].array.tolist()
        tokens = [t.text for t in instances[0]['tokens'].tokens]

        tokens_and_segments = list(zip(tokens, segment_ids))

        expected_tokens_and_segments = [
                 ('[CLS]', 0),
                 ('the', 0),
                 ('big', 1),
                 ('cat', 1),
                 ('##s', 1),
                 ('jumped', 0),
                 ('[UNK]', 0),
                 ('the', 0),
                 ('la', 2),
                 ('##zie', 2),
                 ('##st', 2),
                 ('brown', 2),
                 ('dog', 2),
                 ('##s', 2),
                 ('.', 0),
                 ('[SEP]', 0)
        ]

        self.assertEqual(
                tokens_and_segments,
                expected_tokens_and_segments
        )

    def test_semeval2010_task8_reader_with_entity_markers(self):
        reader_params = Params({
            "type": "semeval2010_task8",
            "entity_masking": "entity_markers",
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
                "bert_model_type": "tests/fixtures/evaluation/semeval2010_task8/vocab_entity_markers.txt",
                "do_lower_case": True,
            },
        })

        reader = DatasetReader.from_params(reader_params)
        train_file = 'tests/fixtures/evaluation/semeval2010_task8/semeval2010_task8.json'

        instances = reader.read(train_file)

        # check that the offsets are right!
        segment_ids = instances[0]['segment_ids'].array.tolist()
        tokens = [t.text for t in instances[0]['tokens'].tokens]

        tokens_and_segments = list(zip(tokens, segment_ids))

        expected_tokens_and_segments = [
             ('[CLS]', 0),
             ('the', 0),
             ('[e1start]', 0),
             ('big', 0),
             ('cat', 0),
             ('##s', 0),
             ('[e1end]', 0),
             ('jumped', 0),
             ('[UNK]', 0),
             ('the', 0),
             ('[e2start]', 0),
             ('la', 0),
             ('##zie', 0),
             ('##st', 0),
             ('brown', 0),
             ('dog', 0),
             ('##s', 0),
             ('[e2end]', 0),
             ('.', 0),
             ('[SEP]', 0)]

        self.assertEqual(
                tokens_and_segments,
                expected_tokens_and_segments
        )

        tokens_1 = [t.text for t in instances[1]['tokens'].tokens]
        expected_tokens_1 = ['[CLS]',
             'the',
             '[e2start]',
             'big',
             'cat',
             '##s',
             '[e2end]',
             'jumped',
             '[e1start]',
             '[UNK]',
             'the',
             'la',
             '##zie',
             '##st',
             'brown',
             'dog',
             '##s',
             '[e1end]',
             '.',
             '[SEP]']

        self.assertEqual(
                tokens_1,
                expected_tokens_1
        )

        self.assertEqual(
                instances[0].fields['label_ids'].label, 0
        )
        self.assertEqual(
                instances[1].fields['label_ids'].label, 8
        )

        all_tokens = [[t.text for t in instances[k]['tokens'].tokens] for k in range(2)]

        for k in range(2):
            self.assertEqual(all_tokens[k][instances[k].fields['index_a'].label], '[e1start]')
            self.assertEqual(all_tokens[k][instances[k].fields['index_b'].label], '[e2start]')


if __name__ == '__main__':
    unittest.main()



