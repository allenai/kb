
import unittest
from kb.include_all import  WicDatasetReader
from allennlp.common import Params
from allennlp.data import DatasetReader, DataIterator, Vocabulary


FIXTURES = 'tests/fixtures/evaluation/wic'


class TestWicReader(unittest.TestCase):
    def test_wic_reader(self):
        reader_params = Params({
            "type": "wic",
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
        instances = reader.read(FIXTURES + '/train')
        iterator = DataIterator.from_params(Params({"type": "basic"}))
        iterator.index_with(Vocabulary())

        for batch in iterator(instances, num_epochs=1, shuffle=False):
            break

        self.assertTrue(len(batch['label_ids']) == 5)

        self.assertEqual(batch['index_a'][0].item(), 3)
        self.assertEqual(batch['index_b'][0].item(), 10)

    def test_wic_reader_entity_markers(self):
        reader_params = Params({
            "type": "wic",
            "entity_markers": True,
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
                "bert_model_type": "tests/fixtures/evaluation/wic/vocab_entity_markers.txt",
                "do_lower_case": True,
            },
        })

        reader = DatasetReader.from_params(reader_params)
        instances = reader.read(FIXTURES + '/train')
        iterator = DataIterator.from_params(Params({"type": "basic"}))
        iterator.index_with(Vocabulary())

        for batch in iterator(instances, num_epochs=1, shuffle=False):
            break

        self.assertTrue(len(batch['label_ids']) == 5)

        self.assertEqual(batch['index_a'][0].item(), 3)
        self.assertEqual(batch['index_b'][0].item(), 12)

        instance_0_text = [token.text for token in instances[0].fields['tokens'].tokens]
        expected_instance_0_text = ['[CLS]', '[UNK]', '[UNK]', '[e1start]', '[UNK]',
            '[e1end]', '[UNK]', '[UNK]', '[UNK]', '.', '[SEP]', '[UNK]', '[e2start]',
            '[UNK]', '[e2end]', '[UNK]', 'over', '[UNK]', '.', '[SEP]'
        ]
        self.assertEqual(instance_0_text, expected_instance_0_text)
        self.assertEqual(instance_0_text[3], '[e1start]')
        self.assertEqual(instance_0_text[12], '[e2start]')


if __name__ == '__main__':
    unittest.main()



