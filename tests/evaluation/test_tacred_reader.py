import unittest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data import DatasetReader
from pytorch_pretrained_bert.tokenization import BertTokenizer

import kb
from kb.bert_tokenizer_and_candidate_generator import BertTokenizerAndCandidateGenerator
from kb.wordnet import WordNetCandidateMentionGenerator
from kb.evaluation.tacred_dataset_reader import LABEL_MAP


def get_reader():
    params = {
        "type": "tacred",
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
            "bert_model_type": "tests/fixtures/tacred/vocab.txt",
            "do_lower_case": True,
        }
    }
    return DatasetReader.from_params(Params(params))


class TestTacredDatasetReader(unittest.TestCase):
    def test_tacred_dataset_reader(self):
        reader = get_reader()
        instances = ensure_list(reader.read('tests/fixtures/tacred/LDC2018T24.json'))

        # Check number of instances is correct
        self.assertEqual(len(instances), 3)

        # Check that first instance's tokens are correct
        tokens_0 = [x.text for x in instances[0]['tokens']]

        initial_tokens_0 = tokens_0[:6]
        expected_initial_tokens_0 = ['[CLS]', 'douglas', 'flint', '[SEP]', 'chairman', '[SEP]']
        self.assertListEqual(initial_tokens_0, expected_initial_tokens_0)

        final_tokens_0 = tokens_0[-6:]
        expected_final_tokens_0 = ['a', 'govern', '##ment', '[UNK]', '.', '[SEP]']
        self.assertListEqual(final_tokens_0, expected_final_tokens_0)

        # Check that first instances label is correct
        label_0 = instances[0]['label_ids'].label
        expected_label_0 = LABEL_MAP['per:title']
        self.assertEqual(label_0, expected_label_0)

    def test_entity_mask(self):
        # Check 'mask' mode has expected behavior
        reader = get_reader()
        reader.entity_masking = 'mask'
        instances = ensure_list(reader.read('tests/fixtures/tacred/LDC2018T24.json'))

        tokens_0 = [x.text for x in instances[0]['tokens']]
        subj_tokens_0 = tokens_0[14]
        self.assertEqual(subj_tokens_0, '[MASK]')

        tokens_0 = [x.text for x in instances[0]['tokens']]
        obj_tokens_0 = tokens_0[17]
        self.assertEqual(obj_tokens_0, '[MASK]')

        # Check 'type/role' mode has expected behavior
        reader.entity_masking = 'type/role'
        instances = ensure_list(reader.read('tests/fixtures/tacred/LDC2018T24.json'))

        tokens_0 = [x.text for x in instances[0]['tokens']]
        subj_tokens_0 = tokens_0[14]
        self.assertEqual(subj_tokens_0, '[s-person]')

        tokens_0 = [x.text for x in instances[0]['tokens']]
        obj_tokens_0 = tokens_0[17]
        self.assertEqual(obj_tokens_0, '[o-title]')

    def test_entity_markers(self):
        reader = get_reader()
        reader.entity_masking = 'entity_markers'

        instances = reader.read('tests/fixtures/tacred/LDC2018T24.json')

        expected_instance_0 = [('[CLS]', 0),
                ('at', 0),
                 ('the', 0),
                 ('same', 0),
                 ('time', 0),
                 (',', 0),
                 ('chief', 0),
                 ('financial', 0),
                 ('[UNK]', 0),
                 ('[e1start]', 0),
                 ('douglas', 0),
                 ('flint', 0),
                 ('[e1end]', 0),
                 ('will', 0),
                 ('become', 0),
                 ('[e2start]', 0),
                 ('chairman', 0),
                 ('[e2end]', 0),
                 (',', 0),
                 ('succeed', 0),
                 ('##ing', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('who', 0),
                 ('is', 0),
                 ('leav', 0),
                 ('##ing', 0),
                 ('to', 0),
                 ('take', 0),
                 ('a', 0),
                 ('govern', 0),
                 ('##ment', 0),
                 ('[UNK]', 0),
                 ('.', 0),
                 ('[SEP]', 0)]

        expected_instance_1 = [('[CLS]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[e2start]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[e2end]', 0),
                 ('in', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('the', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[e1start]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[e1end]', 0),
                 ('[UNK]', 0),
                 ('the', 0),
                 ('[UNK]', 0),
                 ('of', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('by', 0),
                 ('a', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('.', 0),
                 ('[SEP]', 0)]

        for i, expected_instance in zip(range(2), [expected_instance_0, expected_instance_1]):
            instance = list(
                zip([t.text for t in instances[i]['tokens'].tokens],
                     instances[i]['segment_ids'].array.tolist())
            )
            self.assertEqual(instance, expected_instance)
            self.assertEqual(instance[instances[i]['index_a'].label], ('[e1start]', 0))
            self.assertEqual(instance[instances[i]['index_b'].label], ('[e2start]', 0))

    def test_entity_markers_type(self):
        reader = get_reader()
        reader.entity_masking = 'entity_markers/type'

        instances = reader.read('tests/fixtures/tacred/LDC2018T24.json')

        expected_instance_0 = [('[CLS]', 0),
                 ('at', 0),
                 ('the', 0),
                 ('same', 0),
                 ('time', 0),
                 (',', 0),
                 ('chief', 0),
                 ('financial', 0),
                 ('[UNK]', 0),
                 ('[e1start]', 0),
                 ('douglas', 0),
                 ('flint', 0),
                 ('[e1end]', 0),
                 ('will', 0),
                 ('become', 0),
                 ('[e2start]', 0),
                 ('chairman', 0),
                 ('[e2end]', 0),
                 (',', 0),
                 ('succeed', 0),
                 ('##ing', 0),
                 ('[UNK]', 0),
                 ('[UNK]', 0),
                 ('who', 0),
                 ('is', 0),
                 ('leav', 0),
                 ('##ing', 0),
                 ('to', 0),
                 ('take', 0),
                 ('a', 0),
                 ('govern', 0),
                 ('##ment', 0),
                 ('[UNK]', 0),
                 ('.', 0),
                 ('[SEP]', 0),
                 ('[s-person]', 0),
                 ('[SEP]', 0),
                 ('[o-title]', 0),
                 ('[SEP]', 0)
            ]

        instance = list(
            zip([t.text for t in instances[0]['tokens'].tokens],
                 instances[0]['segment_ids'].array.tolist())
        )
        self.assertEqual(instance, expected_instance_0)
        self.assertEqual(instance[instances[0]['index_a'].label], ('[e1start]', 0))
        self.assertEqual(instance[instances[0]['index_b'].label], ('[e2start]', 0))



    def test_entity_role_segment(self):
        reader = get_reader()
        reader.entity_masking = 'type/role/segment'

        instances = reader.read('tests/fixtures/tacred/LDC2018T24.json')

        instance_0 = list(
                zip([t.text for t in instances[0]['tokens'].tokens],
                     instances[0]['segment_ids'].array.tolist())
        )
        expected_instance_0 = [
             ('[CLS]', 0),
             ('at', 0),
             ('the', 0),
             ('same', 0),
             ('time', 0),
             (',', 0),
             ('chief', 0),
             ('financial', 0),
             ('[UNK]', 0),
             ('douglas', 1),
             ('flint', 1),
             ('will', 0),
             ('become', 0),
             ('chairman', 2),
             (',', 0),
             ('succeed', 0),
             ('##ing', 0),
             ('[UNK]', 0),
             ('[UNK]', 0),
             ('who', 0),
             ('is', 0),
             ('leav', 0),
             ('##ing', 0),
             ('to', 0),
             ('take', 0),
             ('a', 0),
             ('govern', 0),
             ('##ment', 0),
             ('[UNK]', 0),
             ('.', 0),
             ('[SEP]', 0),
             ('[s-person]', 1),
             ('[SEP]', 1),
             ('[o-title]', 2),
             ('[SEP]', 2)
        ]
        self.assertEqual(instance_0, expected_instance_0)


if __name__ == '__main__':
    unittest.main()
