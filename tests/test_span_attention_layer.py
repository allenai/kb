
from pytorch_pretrained_bert.modeling import BertConfig
import json

from kb.span_attention_layer import SpanAttentionLayer, SpanWordAttention

import unittest

import torch

class TestSpanAttentionLayer(unittest.TestCase):
    def test_span_word_attention(self):
        config_file = 'tests/fixtures/bert/bert_config.json'
        with open(config_file) as fin:
            json_config = json.load(fin)

        vocab_size = json_config.pop("vocab_size")
        config = BertConfig(vocab_size, **json_config)

        span_attn = SpanWordAttention(config)

        batch_size = 7
        timesteps = 29
        hidden_states = torch.rand(batch_size, timesteps, config.hidden_size)

        num_entity_embeddings = 11
        entity_embeddings = torch.rand(batch_size, num_entity_embeddings, config.hidden_size)
        entity_mask = entity_embeddings[:, :, 0] > 0.5

        span_attn, attention_probs = span_attn(hidden_states, entity_embeddings, entity_mask)
        self.assertEqual(list(span_attn.shape), [batch_size, timesteps, config.hidden_size])

    def test_span_attention_layer(self):
        config_file = 'tests/fixtures/bert/bert_config.json'
        with open(config_file) as fin:
            json_config = json.load(fin)

        vocab_size = json_config.pop("vocab_size")
        config = BertConfig(vocab_size, **json_config)
    
        batch_size = 7
        timesteps = 29
        hidden_states = torch.rand(batch_size, timesteps, config.hidden_size)
    
        num_entity_embeddings = 11
        entity_embeddings = torch.rand(batch_size, num_entity_embeddings, config.hidden_size)
        entity_mask = entity_embeddings[:, :, 0] > 0.5
    
        span_attention_layer = SpanAttentionLayer(config)
    
        output = span_attention_layer(hidden_states, entity_embeddings, entity_mask)

        self.assertEqual(list(output["output"].shape), [batch_size, timesteps, config.hidden_size])


if __name__ == '__main__':
    unittest.main()

