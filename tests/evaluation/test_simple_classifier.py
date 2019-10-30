
import unittest
from kb.include_all import SimpleClassifier, F1Metric
from allennlp.common import Params
from allennlp.models import Model
from allennlp.data import DatasetReader, DataIterator, Vocabulary
from allennlp.training.metrics import CategoricalAccuracy

def get_wic_batch():
    fixtures = 'tests/fixtures/evaluation/wic'

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
    instances = reader.read(fixtures + '/train')
    iterator = DataIterator.from_params(Params({"type": "basic"}))
    iterator.index_with(Vocabulary())

    for batch in iterator(instances, num_epochs=1, shuffle=False):
        break

    return batch


def get_ultra_fine_batch():
    from kb.include_all import UltraFineReader

    params = {
        "type": "ultra_fine",
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
        }
    }

    reader = DatasetReader.from_params(Params(params))
    instances = reader.read('tests/fixtures/evaluation/ultra_fine/train.json')
    iterator = DataIterator.from_params(Params({"type": "basic"}))
    iterator.index_with(Vocabulary())

    for batch in iterator(instances, num_epochs=1, shuffle=False):
        break

    return batch


def get_knowbert_model():
    vocab = Vocabulary.from_params(Params({
        "directory_path": "tests/fixtures/kg_embeddings/tucker_wordnet/vocabulary",
    }))

    params = Params({
        "type": "knowbert",
        "soldered_kgs": {
            "wordnet": {
                "type": "soldered_kg",
                "entity_linker": {
                    "type": "entity_linking_with_candidate_mentions",
                    "kg_model": {
                        "type": "from_archive",
                        "archive_file": "tests/fixtures/kg_embeddings/tucker_wordnet/model.tar.gz",
                    },
                    "contextual_embedding_dim": 12,
                    "max_sequence_length": 64,
                    "span_encoder_config": {
                        "hidden_size": 24,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 3,
                        "intermediate_size": 37
                    },
                },
                "span_attention_config": {
                    "hidden_size": 24,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "intermediate_size": 55
                }
            },
        },
        "soldered_layers": {"wordnet": 1},
        "bert_model_name": "tests/fixtures/bert/bert_test_fixture.tar.gz",
    })

    model = Model.from_params(params, vocab=vocab)
    return model, vocab


class TestSimpleClassifier(unittest.TestCase):
    def test_simple_classifier(self):
        batch = get_wic_batch()
        knowbert_model, vocab = get_knowbert_model()

        model = SimpleClassifier(
            vocab,
            knowbert_model,
            'classification',
            2,
            12,
            CategoricalAccuracy()
        )
        output = model(**batch)
        output['loss'].backward()

        self.assertTrue(True)

    def test_simple_classifier_with_concat_a_b(self):
        batch = get_wic_batch()
        knowbert_model, vocab = get_knowbert_model()

        model = SimpleClassifier(
            vocab,
            knowbert_model,
            'classification',
            2,
            12,
            CategoricalAccuracy(),
            concat_word_a_b=True
        )

        output = model(**batch)
        output['loss'].backward()

        self.assertTrue(True)

    def test_simple_classifier_bce_loss(self):
        batch = get_ultra_fine_batch()
        knowbert_model, vocab = get_knowbert_model()

        model = SimpleClassifier(
            vocab,
            knowbert_model,
            'classification',
            9, # 9 labels
            12,
            F1Metric(),
            use_bce_loss=True
        )

        output = model(**batch)
        output['loss'].backward()

        metrics = model.get_metrics()
        self.assertTrue('f1' in metrics)


if __name__ == '__main__':
    unittest.main()



