

import unittest

import torch

from allennlp.common import Params
from allennlp.data import Vocabulary, DatasetReader
from allennlp.data.iterators import BasicIterator
from allennlp.models import Model


from kb.include_all import ModelArchiveFromParams
from kb.kg_embedding import KGTupleReader

from kb.knowbert import EntityDisambiguator, KnowBert
from kb.knowbert import EntityLinkingWithCandidateMentions, SolderedKG

ARCHIVE_FILE = "tests/fixtures/kg_embeddings/tucker_wordnet/model.tar.gz"


def get_fixtures(include_gold_entities=False,
                 include_lm_labels=True,
                 include_contextual_embeddings=False):
    vocab = Vocabulary.from_params(Params({
        "directory_path": "tests/fixtures/kg_embeddings/tucker_wordnet/vocabulary",
    }))
    
    batch = {'next_sentence_label': torch.tensor([0, 1, 1]),
     'tokens': {'tokens': torch.tensor([[16, 16, 11,  1,  1,  1, 17,  1,  1,  1],
              [16, 16,  1, 12,  1, 17,  1,  1,  1,  1],
              [16, 16,  1,  1, 17,  1, 13, 17, 17,  0]])},
    
     'segment_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]]),

     'lm_label_ids': {'lm_labels': torch.tensor(
            [[0,  1, 0, 0, 13, 0,  1,  1, 13, 0],
             [0, 0,  1, 0, 0,  2,  1,  1, 13, 0],
             [0,  1,  1, 0,  1,  1, 0,  0,  0,  0]])},

     'candidates': {'wordnet': {'candidate_entity_priors': torch.tensor(
              [[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
    
               [[0.2500, 0.2500, 0.2500, 0.2500, 0.0000],
                [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]],
    
               [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]),

       'candidate_entities': {'ids': torch.tensor(
               [[[ 67,   0,   0,   0,   0],
                 [  0,   0,   0,   0,   0]],
    
                [[344, 349, 354, 122,   0],
                 [101,  46, 445,  25,  28]],
    
                [[  0,   0,   0,   0,   0],
                 [  0,   0,   0,   0,   0]]])},

       'candidate_segment_ids': torch.tensor([[0, 1], [0, 1], [0, 0]]),

       'candidate_spans': torch.tensor(
              [[[ 1,  1],
                [-1, -1]],
    
               [[ 1,  1],
                [ 4,  4]],
    
               [[-1, -1],
                [-1, -1]]])}}}

    if include_gold_entities:
        batch['gold_entities'] = {
            'wordnet': {'ids': torch.tensor(
              [[[ 67],
                [0]],

               [[ 349],
                [ 46]],

               [[0],
                [0]]])}}

    if not include_lm_labels:
        del batch['next_sentence_label']
        del batch['lm_label_ids']

    if include_contextual_embeddings:
        batch_size, timesteps = batch['tokens']['tokens'].shape
        batch['contextual_embeddings'] = torch.rand(batch_size, timesteps, 12)
        batch['tokens_mask'] = batch['tokens']['tokens'] > 0
        del batch['tokens']

    return vocab, batch


class TestEntityDisambiguator(unittest.TestCase):
    def test_entity_disambiguator(self):
        vocab, batch = get_fixtures()

        contextual_embedding_dim = 12
        entity_embedding_dim = 11
        entity_embeddings = torch.nn.Embedding(500, entity_embedding_dim)

        disambiguator = EntityDisambiguator(
            contextual_embedding_dim,
            entity_embedding_dim,
            entity_embeddings,
            weighted_entity_threshold=0.3,
            null_entity_id=499
        )

        batch_size, timesteps = batch['tokens']['tokens'].shape
        contextual_embeddings = torch.rand(batch_size, timesteps, contextual_embedding_dim)

        output = disambiguator(
            contextual_embeddings,
            batch['tokens']['tokens'] > 0,
            batch['candidates']['wordnet']['candidate_spans'],
            batch['candidates']['wordnet']['candidate_entities']['ids'],
            batch['candidates']['wordnet']['candidate_entity_priors'],
            batch['candidates']['wordnet']['candidate_segment_ids']
        )

        _, num_spans, num_candidates = batch['candidates']['wordnet']['candidate_entities']['ids'].shape
        self.assertEqual(
            list(output['linking_scores'].shape), [batch_size, num_spans, num_candidates]
        )
        self.assertEqual(
            list(output['weighted_entity_embeddings'].shape), [batch_size, num_spans, entity_embedding_dim]
        )


class TestEntityLinkingWithCandidateMentions(unittest.TestCase):
    def _get_model(self, vocab):
        params = Params({
            "type": "entity_linking_with_candidate_mentions",
            "kg_model": {
                "type": "from_archive",
                "archive_file": ARCHIVE_FILE,
            },
            "contextual_embedding_dim": 12,
        })
        model = Model.from_params(params, vocab=vocab)
        model.unfreeze(None)
        return model


    def test_entity_linking(self):
        vocab, batch = get_fixtures(include_gold_entities=True,
                                    include_lm_labels=False,
                                    include_contextual_embeddings=True)
        model = self._get_model(vocab)

        output = model(
            batch['contextual_embeddings'],
            batch['tokens_mask'],
            gold_entities=batch['gold_entities']['wordnet'],
            **batch['candidates']['wordnet'])

        self.assertTrue('loss' in output)


class TestSolderedKG(unittest.TestCase):
    def test_soldered_kg(self):
        vocab, batch = get_fixtures(include_gold_entities=True,
                                    include_lm_labels=False,
                                    include_contextual_embeddings=True)

        params = Params({
            "type": "soldered_kg",
            "entity_linker": {
                "type": "entity_linking_with_candidate_mentions",
                "kg_model": {
                    "type": "from_archive",
                    "archive_file": ARCHIVE_FILE,
                },
                "contextual_embedding_dim": 12,
            },
            "span_attention_config": {
                "hidden_size": 24,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 55
            }
        })
        model = Model.from_params(params, vocab=vocab)
        model.unfreeze(None)

        output = model(
                batch['contextual_embeddings'],
                batch['tokens_mask'],
                **batch['candidates']['wordnet']
        )

        self.assertEqual(
            batch['contextual_embeddings'].shape,
            output['contextual_embeddings'].shape
        )


def get_knowbert(vocab, mode, include_wiki=False):
    params = {
       "type": "knowbert",
        "mode": mode,
        "soldered_kgs": {
            "wordnet": {
                "type": "soldered_kg",
                "entity_linker": {
                    "type": "entity_linking_with_candidate_mentions",
                    "kg_model": {
                        "type": "from_archive",
                        "archive_file": ARCHIVE_FILE,
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
    }

    if include_wiki:
        params["soldered_kgs"]["wiki"] = {
                "type": "soldered_kg",
                "entity_linker": {
                    "type": "entity_linking_with_candidate_mentions",
                    "namespace": "entity_wiki",
                    "entity_embedding": {
                        "num_embeddings": 14,
                        "embedding_dim": 24,
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
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "intermediate_size": 55
                }
            }
        params["soldered_layers"]["wiki"] = 0
        params["soldered_kgs"]["wordnet"]["entity_linker"]["namespace"] = "entity_wordnet"

    model = Model.from_params(Params(params), vocab=vocab)
    return model


class TestKnowBert(unittest.TestCase):
    def test_knowbert_el(self):
        vocab, batch = get_fixtures(include_gold_entities=True)
        model = get_knowbert(vocab, None)
        output = model(**batch)
        loss = output['loss']
        loss.backward()
        self.assertTrue(True)

    def test_knowbert_el_pretrain(self):
        vocab, batch = get_fixtures(include_gold_entities=True)
        model = get_knowbert(vocab, 'entity_linking')
        output = model(**batch)
        loss = output['loss']
        loss.backward()
        self.assertTrue(True)

    def test_knowbert_simple(self):
        vocab, batch = get_fixtures()
        model = get_knowbert(vocab, None)
        output = model(**batch)
        loss = output['loss']
        loss.backward()
        self.assertTrue(True)


class TestKnowBertWikiWordnet(unittest.TestCase):
    def test_knowbert_wiki_wordnet(self):
        from kb.testing import get_bert_pretraining_reader_with_kg

        reader = get_bert_pretraining_reader_with_kg(
            mask_candidate_strategy='full_mask', masked_lm_prob=0.35, include_wiki=True)
        instances = reader.read("tests/fixtures/bert_pretraining/shard1.txt")

        vocab = Vocabulary.from_params(Params({
            "directory_path": "tests/fixtures/wordnet_wiki_vocab",
        }))

        iterator = BasicIterator()
        iterator.index_with(vocab)

        for batch in iterator(instances, num_epochs=1, shuffle=False):
            pass

        # hack, incompatitable fixtures...
        batch['tokens']['tokens'] = torch.min(batch['tokens']['tokens'], torch.tensor([17]))
        batch['lm_label_ids']['lm_labels'] = torch.min(batch['lm_label_ids']['lm_labels'], torch.tensor([17]))
        model = get_knowbert(vocab, None, include_wiki=True)
        output = model(**batch)
        loss = output['loss']
        loss.backward()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()

