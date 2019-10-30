
import unittest

from allennlp.common.params import Params
from allennlp.data.dataset_readers import CcgBankDatasetReader
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset_readers import Conll2003DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.data import Vocabulary

from kb.multitask import MultitaskDatasetReader, MultiTaskDataIterator


FIXTURES_ROOT = 'tests/fixtures/multitask'


def get_dataset_params_paths(datasets_for_vocab_creation):
    params = Params({
        "type": "multitask_reader",
        "dataset_readers": {
            "ner": {
                "type": "conll2003",
                "tag_label": "ner",
                "token_indexers": {
                    "tokens": {
                        "type": "single_id",
                    },
                }
            },
            "ccg": {
                "type": "ccgbank",
                "token_indexers": {
                    "tokens": {
                        "type": "single_id",
                    },
                },
                "feature_labels": ["original_pos"],
            }
        },
        "datasets_for_vocab_creation": datasets_for_vocab_creation
    })

    file_paths = {
        'ner': FIXTURES_ROOT + '/conll2003.txt',
        'ccg': FIXTURES_ROOT + '/ccgbank.txt'
    }

    return params, file_paths


class TestMultiTaskDatasetReader(unittest.TestCase):
    def test_read(self):
        params, file_paths = get_dataset_params_paths(["ner"])

        multitask_reader = DatasetReader.from_params(params)

        dataset = multitask_reader.read(file_paths)

        # get all the instances --  only should have "original_pos_tags"
        # for NER
        for name, instances in dataset.datasets.items():
            self.assertTrue(name in ('ner', 'ccg'))
            for instance in instances:
                if name == 'ner':
                    self.assertTrue("original_pos_tags" not in instance.fields)
                else:
                    self.assertTrue("original_pos_tags" in instance.fields)

        # when iterating directly, only get 'ner'
        for instance in dataset:
            self.assertTrue("original_pos_tags" not in instance.fields)


class TestMultiTaskDataIterator(unittest.TestCase):
    def test_multi_iterator(self):
        params, file_paths = get_dataset_params_paths(['ner', 'ccg'])

        multitask_reader = DatasetReader.from_params(params)
        dataset = multitask_reader.read(file_paths)

        iterator_params = Params({
            "type": "multitask_iterator",
            "iterators": {
                "ner": {"type": "bucket",
                        "sorting_keys": [["tokens", "num_tokens"]],
                        "padding_noise": 0.0,
                        "batch_size" : 2},
                "ccg": {"type": "basic",
                        "batch_size" : 1}
            },
            "names_to_index": ["ner", "ccg"],
        })

        multi_iterator = DataIterator.from_params(iterator_params)

        # make the vocab
        vocab = Vocabulary.from_params(Params({}),
                                       (instance for instance in dataset))
        multi_iterator.index_with(vocab)

        all_batches = []
        for epoch in range(2):
            all_batches.append([])
            for batch in multi_iterator(dataset, shuffle=True,
                                        num_epochs=1):
                all_batches[-1].append(batch)

        # 3 batches per epoch -
        self.assertEqual([len(b) for b in all_batches], [3, 3])

        ner_batches = []
        ccg_batches = []
        for epoch_batches in all_batches:
            ner_batches.append(0)
            ccg_batches.append(0)
            for batch in epoch_batches:
                if 'original_pos_tags' not in batch:
                    ner_batches[-1] += 1
                if 'original_pos_tags' in batch:
                    ccg_batches[-1] += 1

        # 1 NER batch per epoch, 2 CCG per epoch
        self.assertEqual(ner_batches, [1, 1])
        self.assertEqual(ccg_batches, [2, 2])


if __name__ == '__main__':
    unittest.main()

