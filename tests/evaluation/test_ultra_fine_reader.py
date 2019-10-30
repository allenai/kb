import unittest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data import DatasetReader, DataIterator, Vocabulary

from kb.include_all import UltraFineReader


def get_reader(entity_masking):
    params = {
        "type": "ultra_fine",
        "entity_masking": entity_masking,
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
            "bert_model_type": "tests/fixtures/evaluation/ultra_fine/vocab.txt",
            "do_lower_case": True,
        }
    }
    return DatasetReader.from_params(Params(params))


class TestUltraFineReader(unittest.TestCase):
    def test_ultra_fine_reader(self):
        reader = get_reader("entity")
        instances = ensure_list(reader.read('tests/fixtures/evaluation/ultra_fine/train.json'))

        # Check number of instances is correct
        self.assertEqual(len(instances), 2)

        # Check that first instance's tokens are correct
        tokens_0 = [x.text for x in instances[0]['tokens']]
        segments_0 = list(instances[0]['segment_ids'].array)
        actual = list(zip(tokens_0, segments_0))
        expected = [('[CLS]', 0),
                 ('the', 0),
                 ('british', 0),
                 ('information', 0),
                 ('commissioner', 0),
                 ("'s", 0),
                 ('office', 0),
                 ('invites', 0),
                 ('[unused0]', 0),
                 ('to', 0),
                 ('locate', 0),
                 ('its', 0),
                 ('add', 0),
                 ('##ress', 0),
                 ('using', 0),
                 ('google', 0),
                 ('[UNK]', 0),
                 ('.', 0),
                 ('[SEP]', 0),
                 ('web', 1),
                 ('users', 1),
                 ('[SEP]', 1)]
        self.assertListEqual(actual, expected)

        iterator = DataIterator.from_params(Params({"type": "basic"}))
        iterator.index_with(Vocabulary())

        for batch in iterator(instances, num_epochs=1, shuffle=False):
            break

        expected_labels = [[0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertEqual(batch['label_ids'].numpy().tolist(), expected_labels)

    def test_ultra_fine_reader_entity_markers(self):
        reader = get_reader("entity_markers")
        instances = ensure_list(reader.read('tests/fixtures/evaluation/ultra_fine/train.json'))

        # Check number of instances is correct
        self.assertEqual(len(instances), 2)

        # Check that first instance's tokens are correct
        tokens_0 = [x.text for x in instances[0]['tokens']]
        segments_0 = list(instances[0]['segment_ids'].array)
        actual = list(zip(tokens_0, segments_0))
        expected = [('[CLS]', 0),
                 ('the', 0),
                 ('british', 0),
                 ('information', 0),
                 ('commissioner', 0),
                 ("'s", 0),
                 ('office', 0),
                 ('invites', 0),
                 ('[e1start]', 0),
                 ('web', 0),
                 ('users', 0),
                 ('[e1end]', 0),
                 ('to', 0),
                 ('locate', 0),
                 ('its', 0),
                 ('add', 0),
                 ('##ress', 0),
                 ('using', 0),
                 ('google', 0),
                 ('[UNK]', 0),
                 ('.', 0),
                 ('[SEP]', 0)]
        self.assertListEqual(actual, expected)

        self.assertEqual(actual[instances[0]['index_a'].label], ('[e1start]', 0))


if __name__ == '__main__':
    unittest.main()
