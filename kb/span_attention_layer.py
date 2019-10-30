
import torch
import math

from pytorch_pretrained_bert.modeling import BertIntermediate, BertOutput, BertLayer, BertSelfOutput

from kb.common import get_dtype_for_module, extend_attention_mask_for_bert, get_linear_layer_init_identity, init_bert_weights


class SpanWordAttention(torch.nn.Module):
    def __init__(self, config):
        super(SpanWordAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        #self.query = get_linear_layer_init_identity(config.hidden_size)
        #self.key = get_linear_layer_init_identity(config.hidden_size)
        #self.value = get_linear_layer_init_identity(config.hidden_size)

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, entity_embeddings, entity_mask):
        """
        hidden_states = (batch_size, timesteps, dim)
        entity_embeddings = (batch_size, num_entities, dim)
        entity_mask = (batch_size, num_entities) with 0/1
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(entity_embeddings)
        mixed_value_layer = self.value(entity_embeddings)

        # (batch_size, num_heads, timesteps, head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # (batch_size, num_heads, num_entity_embeddings, head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch_size, num_heads, timesteps, num_entity_embeddings)
        # gives the attention from timestep i to embedding j
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # apply the attention mask.
        # the attention_mask masks out thing to attend TO so we extend
        # the entity mask
        attention_mask = extend_attention_mask_for_bert(entity_mask, get_dtype_for_module(self))
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire entities to attend to, which might
        # seem a bit unusual, but is similar to the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # (batch_size, num_heads, timesteps, head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (batch_size, timesteps, num_heads, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # (batch_size, timesteps, hidden_dim)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class SpanAttention(torch.nn.Module):
    def __init__(self, config):
        super(SpanAttention, self).__init__()
        self.attention = SpanWordAttention(config)
        init_bert_weights(self.attention, config.initializer_range, (SpanWordAttention, ))
        self.output = BertSelfOutput(config)
        init_bert_weights(self.output, config.initializer_range)

    def forward(self, input_tensor, entity_embeddings, entity_mask):
        span_output, attention_probs = self.attention(input_tensor, entity_embeddings, entity_mask)
        attention_output = self.output(span_output, input_tensor)
        return attention_output, attention_probs


class SpanAttentionLayer(torch.nn.Module):
    # WARNING: does it's own init, so don't re-init
    def __init__(self, config):
        super(SpanAttentionLayer, self).__init__()
        self.attention = SpanAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        init_bert_weights(self.intermediate, config.initializer_range)
        init_bert_weights(self.output, config.initializer_range)

    def forward(self, hidden_states, entity_embeddings, entity_mask):
        attention_output, attention_probs = self.attention(hidden_states, entity_embeddings, entity_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return {"output": layer_output, "attention_probs": attention_probs}

