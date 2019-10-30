from typing import Dict, List

import math
import tarfile
import re

from allennlp.models import Model
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.data import Vocabulary
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.training.metrics import Average, CategoricalAccuracy
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import device_mapping
from allennlp.common.file_utils import cached_path


from pytorch_pretrained_bert.modeling import BertForPreTraining, BertLayer, BertLayerNorm, BertConfig, BertEncoder

import torch
import numpy as np

from kb.metrics import MeanReciprocalRank
from kb.entity_linking import BaseEntityDisambiguator, EntityLinkingBase
from kb.span_attention_layer import SpanAttentionLayer
from kb.common import get_dtype_for_module, extend_attention_mask_for_bert, init_bert_weights
from kb.common import EntityEmbedder, set_requires_grad
from kb.evaluation.exponential_average_metric import ExponentialMovingAverage
from kb.evaluation.weighted_average import WeightedAverage


def print_shapes(x, prefix='', raise_on_nan=False):
    if isinstance(x, torch.Tensor):
        print(prefix, x.shape)
        if x.dtype == torch.float32 or x.dtype == torch.float16:
            print(x.min(), x.max(), x.mean(), x.std())
        if raise_on_nan and torch.isnan(x).long().sum().item() > 0:
            print("GOT NAN!!")
            raise ValueError
    elif isinstance(x, (list, tuple)):
        for ele in x:
            print_shapes(ele, prefix + '-->')
    elif isinstance(x, dict):
        for k, v in x.items():
            print_shapes(v, prefix + ' ' + k + ':')
    else:
        print("COULDN'T get shape ", type(x))
            
def diagnose_backward_hook(module, m_input, m_output):
    print("------")
    print('Inside ' + module.__class__.__name__ + ' backward')
    print('Inside class:' + module.__class__.__name__)
    print("INPUT:")
    print_shapes(m_input)
    print("OUTPUT:")
    print_shapes(m_output)
    print("=======")

def diagnose_forward_hook(module, m_input, m_output):
    print("------")
    print('Inside ' + module.__class__.__name__ + ' forward')
    print('Inside class:' + module.__class__.__name__)
    print("INPUT:")
    print_shapes(m_input)
    print("OUTPUT:")
    print_shapes(m_output, raise_on_nan=True)
    print("=======")


class BertPretrainedMetricsLoss(Model):
    def __init__(self, vocab: Vocabulary,
                       regularizer: RegularizerApplicator = None):
        super().__init__(vocab, regularizer)

        self.nsp_loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.lm_loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)

        self._metrics = {
            "total_loss_ema": ExponentialMovingAverage(alpha=0.5),
            "nsp_loss_ema": ExponentialMovingAverage(alpha=0.5),
            "lm_loss_ema": ExponentialMovingAverage(alpha=0.5),
            "total_loss": Average(),
            "nsp_loss": Average(),
            "lm_loss": Average(),
            "lm_loss_wgt": WeightedAverage(),
            "mrr": MeanReciprocalRank(),
        }
        self._accuracy = CategoricalAccuracy()

    def get_metrics(self, reset: bool = False):
        metrics = {k: v.get_metric(reset) for k, v in self._metrics.items()}
        metrics['nsp_accuracy'] = self._accuracy.get_metric(reset)
        return metrics

    def _compute_loss(self,
                      contextual_embeddings,
                      pooled_output,
                      lm_label_ids,
                      next_sentence_label,
                      update_metrics=True):

        # (batch_size, timesteps, vocab_size), (batch_size, 2)
        prediction_scores, seq_relationship_score = self.pretraining_heads(
                contextual_embeddings, pooled_output
        )

        loss_metrics = []
        if lm_label_ids is not None:
            # Loss
            vocab_size = prediction_scores.shape[-1]
            masked_lm_loss = self.lm_loss_function(
                prediction_scores.view(-1, vocab_size), lm_label_ids["lm_labels"].view(-1)
            )
            masked_lm_loss_item = masked_lm_loss.item()
            loss_metrics.append([["lm_loss_ema", "lm_loss"], masked_lm_loss_item])
            num_lm_predictions = (lm_label_ids["lm_labels"] > 0).long().sum().item()
            self._metrics['lm_loss_wgt'](masked_lm_loss_item, num_lm_predictions)
        else:
            masked_lm_loss = 0.0

        if next_sentence_label is not None:
            next_sentence_loss = self.nsp_loss_function(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            loss_metrics.append([["nsp_loss_ema", "nsp_loss"], next_sentence_loss.item()])
            if update_metrics:
                self._accuracy(
                    seq_relationship_score.detach(), next_sentence_label.view(-1).detach()
                )
        else:
            next_sentence_loss = 0.0

        # update metrics
        if update_metrics:
            total_loss = masked_lm_loss + next_sentence_loss
            for keys, v in [[["total_loss_ema", "total_loss"], total_loss.item()]] + loss_metrics:
                for key in keys:
                    self._metrics[key](v)

        return masked_lm_loss, next_sentence_loss

    def _compute_mrr(self,
                     contextual_embeddings,
                     pooled_output,
                     lm_labels_ids,
                     mask_indicator):
        prediction_scores, seq_relationship_score = self.pretraining_heads(
                contextual_embeddings, pooled_output
        )
        self._metrics['mrr'](prediction_scores, lm_labels_ids, mask_indicator)


@Model.register("bert_pretrained_masked_lm")
class BertPretrainedMaskedLM(BertPretrainedMetricsLoss):
    """
    So we can evaluate and compute the loss of the pretrained bert model
    """
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model_name: str,
                 remap_segment_embeddings: int = None,
                 regularizer: RegularizerApplicator = None):
        super().__init__(vocab, regularizer)

        pretrained_bert = BertForPreTraining.from_pretrained(bert_model_name)
        self.pretraining_heads = pretrained_bert.cls
        self.bert = pretrained_bert

        self.remap_segment_embeddings = remap_segment_embeddings
        if remap_segment_embeddings is not None:
            new_embeddings = self._remap_embeddings(self.bert.bert.embeddings.token_type_embeddings.weight)
            if new_embeddings is not None:
                del self.bert.bert.embeddings.token_type_embeddings
                self.bert.bert.embeddings.token_type_embeddings = new_embeddings

    def _remap_embeddings(self, token_type_embeddings):
        embed_dim = token_type_embeddings.shape[1]
        if list(token_type_embeddings.shape) == [self.remap_segment_embeddings, embed_dim]:
            # already remapped!
            return None
        new_embeddings = torch.nn.Embedding(self.remap_segment_embeddings, embed_dim)
        new_embeddings.weight.data.copy_(token_type_embeddings.data[0, :])
        return new_embeddings

    def load_state_dict(self, state_dict, strict=True):
        if self.remap_segment_embeddings:
            # hack the embeddings!
            new_embeddings = self._remap_embeddings(state_dict['bert.bert.embeddings.token_type_embeddings.weight'])
            if new_embeddings is not None:
                state_dict['bert.bert.embeddings.token_type_embeddings.weight'] = new_embeddings.weight
        super().load_state_dict(state_dict, strict=strict)


    def forward(self,
                tokens,
                segment_ids,
                lm_label_ids=None,
                next_sentence_label=None,
                **kwargs):
        mask = tokens['tokens'] > 0
        contextual_embeddings, pooled_output = self.bert.bert(
            tokens['tokens'], segment_ids,
            mask, output_all_encoded_layers=False
        )
        if lm_label_ids is not None or next_sentence_label is not None:
            masked_lm_loss, next_sentence_loss = self._compute_loss(
                contextual_embeddings, pooled_output, lm_label_ids, next_sentence_label
            )
            loss = masked_lm_loss + next_sentence_loss
        else:
            loss = 0.0

        if 'mask_indicator' in kwargs:
            self._compute_mrr(contextual_embeddings,
                              pooled_output,
                              lm_label_ids['lm_labels'],
                              kwargs['mask_indicator'])
        return {'loss': loss,
                'contextual_embeddings': contextual_embeddings,
                'pooled_output': pooled_output}


# KnowBert:
#   Combines bert with one or more SolderedKG
#
#   each SolderedKG is inserted at a particular level, given by an index,
#   such that we run Bert to the index, then the SolderedKG, then the rest
#   of bert.  Indices such that index 0 means run the first contextual layer,
#   then add KG, and index 11 means run to the top of Bert, then the KG
#   (for bert base with 12 layers).
#

class DotAttentionWithPrior(torch.nn.Module):
    def __init__(self,
                 output_feed_forward_hidden_dim: int = 100,
                 weighted_entity_threshold: float = None,
                 null_embedding: torch.Tensor = None,
                 initializer_range: float = 0.02):

        super().__init__()

        # layers for the dot product attention
        self.out_layer_1 = torch.nn.Linear(2, output_feed_forward_hidden_dim)
        self.out_layer_2 = torch.nn.Linear(output_feed_forward_hidden_dim, 1)
        init_bert_weights(self.out_layer_1, initializer_range)
        init_bert_weights(self.out_layer_2, initializer_range)

        self.weighted_entity_threshold = weighted_entity_threshold
        if null_embedding is not None:
            self.register_buffer('null_embedding', null_embedding)

    def forward(self,
            projected_span_representations,
            candidate_entity_embeddings,
            candidate_entity_prior,
            entity_mask):
        """
        projected_span_representations = (batch_size, num_spans, entity_dim)
        candidate_entity_embeddings = (batch_size, num_spans, num_candidates, entity_embedding_dim)
        candidate_entity_prior = (batch_size, num_spans, num_candidates)
            with prior probability of each candidate entity.
            0 <= candidate_entity_prior <= 1 and candidate_entity_prior.sum(dim=-1) == 1
        entity_mask = (batch_size, num_spans, num_candidates)
            with 0/1 bool of whether it is a valid candidate

        returns dict with:
            linking_scores: linking sccore to each entity in each span
                (batch_size, num_spans, num_candidates)
                masked with -10000 for invalid links
            weighted_entity_embeddings: weighted entity embedding
                (batch_size, num_spans, entity_dim)
        """
        # dot product between span embedding and entity embeddings, scaled
        # by sqrt(dimension) as in Transformer
        # (batch_size, num_spans, num_candidates)
        scores = torch.sum(
            projected_span_representations.unsqueeze(-2) * candidate_entity_embeddings,
            dim=-1
        ) / math.sqrt(candidate_entity_embeddings.shape[-1])

        # compute the final score
        # the prior needs to be input as float32 due to half not supported on
        # cpu.  so need to cast it here.
        dtype = list(self.parameters())[0].dtype
        scores_with_prior = torch.cat(
            [scores.unsqueeze(-1), candidate_entity_prior.unsqueeze(-1).to(dtype)],
            dim=-1
        )

        # (batch_size, num_spans, num_candidates)
        linking_score = self.out_layer_2(
            torch.nn.functional.relu(self.out_layer_1(scores_with_prior))
        ).squeeze(-1)

        # mask out the invalid candidates
        invalid_candidate_mask = ~entity_mask

        linking_scores = linking_score.masked_fill(invalid_candidate_mask, -10000.0)
        return_dict = {'linking_scores': linking_scores}

        weighted_entity_embeddings = self._get_weighted_entity_embeddings(
                linking_scores, candidate_entity_embeddings
        )
        return_dict['weighted_entity_embeddings'] = weighted_entity_embeddings

        return return_dict

    def _get_weighted_entity_embeddings(self, linking_scores, candidate_entity_embeddings):
        """
        Get the entity linking weighted entity embedding

        linking_scores = (batch_size, num_spans, num_candidates)
             with unnormalized scores and masked with very small value
            (-10000) for invalid candidates.
        candidate_entity_embeddings = (batch_size, num_spans, num_candidates, entity_embedding_dim)

        returns weighted_entity_embeddings = (batch_size, num_spans, entity_dim)
        """
        # compute softmax of linking scores
        # if we are using the decode threshold, set all scores less then
        # the threshold to small values so they aren't given any weight
        if self.weighted_entity_threshold is not None:
            below_threshold = linking_scores < self.weighted_entity_threshold
            linking_scores = linking_scores.masked_fill(below_threshold, -10000.0)

        # (batch_size, num_spans, num_candidates)
        normed_linking_scores = torch.nn.functional.softmax(linking_scores, dim=-1)

        # use softmax to get weighted entity embedding from candidates
        # (batch_size, num_spans, entity_dim)
        weighted_entity_embeddings = (normed_linking_scores.unsqueeze(-1) * candidate_entity_embeddings).sum(dim=2)

        # if we have a decode threshold, some spans won't have a single
        # predicted entity above the threshold, need to replace them with
        # NULL
        if self.weighted_entity_threshold is not None:
            num_candidates = linking_scores.shape[-1]
            # (batch_size, num_spans)
            all_below_threshold = (below_threshold == 1).long().sum(dim=-1) == num_candidates
            weighted_entity_embeddings[all_below_threshold] = self.null_embedding

        return weighted_entity_embeddings


@BaseEntityDisambiguator.register("diambiguator")
class EntityDisambiguator(BaseEntityDisambiguator, torch.nn.Module):
    def __init__(self,
                 contextual_embedding_dim,
                 entity_embedding_dim: int,
                 entity_embeddings: torch.nn.Embedding,
                 max_sequence_length: int = 512,
                 span_encoder_config: Dict[str, int] = None,
                 dropout: float = 0.1,
                 output_feed_forward_hidden_dim: int = 100,
                 initializer_range: float = 0.02,
                 weighted_entity_threshold: float = None,
                 null_entity_id: int = None,
                 include_null_embedding_in_dot_attention: bool = False):
        """
        Idea: Align the bert and KG vector space by learning a mapping between
            them.
        """
        super().__init__()

        self.span_extractor = SelfAttentiveSpanExtractor(entity_embedding_dim)
        init_bert_weights(self.span_extractor._global_attention._module,
                          initializer_range)

        self.dropout = torch.nn.Dropout(dropout)

        self.bert_to_kg_projector = torch.nn.Linear(
                contextual_embedding_dim, entity_embedding_dim)
        init_bert_weights(self.bert_to_kg_projector, initializer_range)
        self.projected_span_layer_norm = BertLayerNorm(entity_embedding_dim, eps=1e-5)
        init_bert_weights(self.projected_span_layer_norm, initializer_range)

        self.kg_layer_norm = BertLayerNorm(entity_embedding_dim, eps=1e-5)
        init_bert_weights(self.kg_layer_norm, initializer_range)

        # already pretrained, don't init
        self.entity_embeddings = entity_embeddings
        self.entity_embedding_dim = entity_embedding_dim

        # layers for the dot product attention
        if weighted_entity_threshold is not None or include_null_embedding_in_dot_attention:
            if hasattr(self.entity_embeddings, 'get_null_embedding'):
                null_embedding = self.entity_embeddings.get_null_embedding()
            else:
                null_embedding = self.entity_embeddings.weight[null_entity_id, :]
        else:
            null_embedding = None
        self.dot_attention_with_prior = DotAttentionWithPrior(
                 output_feed_forward_hidden_dim,
                 weighted_entity_threshold,
                 null_embedding,
                 initializer_range
        )
        self.null_entity_id = null_entity_id
        self.contextual_embedding_dim = contextual_embedding_dim

        if span_encoder_config is None:
            self.span_encoder = None
        else:
            # create BertConfig
            assert len(span_encoder_config) == 4
            config = BertConfig(
                0, # vocab size, not used
                hidden_size=span_encoder_config['hidden_size'],
                num_hidden_layers=span_encoder_config['num_hidden_layers'],
                num_attention_heads=span_encoder_config['num_attention_heads'],
                intermediate_size=span_encoder_config['intermediate_size']
            )
            self.span_encoder = BertEncoder(config)
            init_bert_weights(self.span_encoder, initializer_range)

    def unfreeze(self, mode):
        def _is_in_alignment(n):
            if 'bert_to_kg_projector' in n:
                return True
            elif 'projected_span_layer_norm' in n:
                return True
            elif 'kg_position_embeddings.embedding_projection' in n:
                return True
            elif 'kg_position_embeddings.position_layer_norm' in n:
                return True
            elif 'kg_layer_norm' in n:
                return True
            elif 'span_extractor' in n:
                return True
            else:
                return False

        if mode == 'entity_linking':
            # learning the entity linker
            for n, p in self.named_parameters():
                if _is_in_alignment(n):
                    p.requires_grad_(True)
                elif 'entity_embeddings.weight' in n:
                    p.requires_grad_(False)
                elif 'kg_position_embeddings' in n:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
        elif mode == 'freeze':
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for n, p in self.named_parameters():
                if 'entity_embeddings.weight' in n:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)

    def _run_span_encoders(self, x, span_mask):
        # run the transformer
        attention_mask = extend_attention_mask_for_bert(span_mask, get_dtype_for_module(self))
        return self.span_encoder(
            x, attention_mask,
            output_all_encoded_layers=False
        )

    def forward(self,
                contextual_embeddings: torch.Tensor,
                mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                **kwargs
        ):
        """
        contextual_embeddings = (batch_size, timesteps, dim) output
            from language model
        mask = (batch_size, num_times)
        candidate_spans = (batch_size, max_num_spans, 2) with candidate
            mention spans. This gives the start / end location for each
            span such span i in row k has:
                start, end = candidate_spans[k, i, :]
                span_embeddings = contextual_embeddings[k, start:end, :]
            it is padded with -1
        candidate_entities = (batch_size, max_num_spans, max_entity_ids)
            padded with 0
        candidate_entity_prior = (batch_size, max_num_spans, max_entity_ids)
            with prior probability of each candidate entity.
            0 <= candidate_entity_prior <= 1 and candidate_entity_prior.sum(dim=-1) == 1

        Returns:
            linking sccore to each entity in each span
                (batch_size, max_num_spans, max_entity_ids)
            masked with -10000 for invalid links
        """
        # get the candidate entity embeddings
        # (batch_size, num_spans, num_candidates, entity_embedding_dim)
        candidate_entity_embeddings = self.entity_embeddings(candidate_entities)
        candidate_entity_embeddings = self.kg_layer_norm(candidate_entity_embeddings.contiguous())

        # project to entity embedding dim
        # (batch_size, timesteps, entity_dim)
        projected_bert_representations = self.bert_to_kg_projector(contextual_embeddings)

        # compute span representations
        span_mask = (candidate_spans[:, :, 0] > -1).long()
        # (batch_size, num_spans, embedding_dim)
        projected_span_representations = self.span_extractor(
            projected_bert_representations,
            candidate_spans,
            mask,
            span_mask
        )
        projected_span_representations = self.projected_span_layer_norm(projected_span_representations.contiguous())

        # run the span transformer encoders
        if self.span_encoder is not None:
            projected_span_representations = self._run_span_encoders(
                projected_span_representations, span_mask
            )[-1]

        entity_mask = candidate_entities > 0
        return_dict = self.dot_attention_with_prior(
                    projected_span_representations,
                    candidate_entity_embeddings,
                    candidate_entity_priors,
                    entity_mask)

        return_dict['projected_span_representations'] = projected_span_representations
        return_dict['projected_bert_representations'] = projected_bert_representations

        return return_dict



@Model.register("entity_linking_with_candidate_mentions")
class EntityLinkingWithCandidateMentions(EntityLinkingBase):
    def __init__(self,
                 vocab: Vocabulary,
                 kg_model: Model = None,
                 entity_embedding: Embedding = None,
                 concat_entity_embedder: EntityEmbedder = None,
                 contextual_embedding_dim: int = None,
                 span_encoder_config: Dict[str, int] = None,
                 margin: float = 0.2,
                 decode_threshold: float = 0.0,
                 loss_type: str = 'margin',
                 max_sequence_length: int = 512,
                 dropout: float = 0.1,
                 output_feed_forward_hidden_dim: int = 100,
                 initializer_range: float = 0.02,
                 include_null_embedding_in_dot_attention: bool = False,
                 namespace: str = 'entity',
                 regularizer: RegularizerApplicator = None):

        super().__init__(vocab,
                         margin=margin,
                         decode_threshold=decode_threshold,
                         loss_type=loss_type,
                         namespace=namespace,
                         regularizer=regularizer)

        num_embeddings_passed = sum(
            [kg_model is not None, entity_embedding is not None, concat_entity_embedder is not None]
        )
        if num_embeddings_passed != 1:
            raise ValueError("Linking model needs either a kg factorisation model or an entity embedding.")

        elif kg_model is not None:
            entity_embedding = kg_model.get_entity_embedding()
            entity_embedding_dim  = entity_embedding.embedding_dim

        elif entity_embedding is not None:
            entity_embedding_dim  = entity_embedding.get_output_dim()

        elif concat_entity_embedder is not None:
            entity_embedding_dim  = concat_entity_embedder.get_output_dim()
            set_requires_grad(concat_entity_embedder, False)
            entity_embedding = concat_entity_embedder

        if loss_type == 'margin':
            weighted_entity_threshold = decode_threshold
        else:
            weighted_entity_threshold = None

        null_entity_id = self.vocab.get_token_index('@@NULL@@', namespace)
        assert null_entity_id != self.vocab.get_token_index('@@UNKNOWN@@', namespace)

        self.disambiguator = EntityDisambiguator(
                 contextual_embedding_dim,
                 entity_embedding_dim=entity_embedding_dim,
                 entity_embeddings=entity_embedding,
                 max_sequence_length=max_sequence_length,
                 span_encoder_config=span_encoder_config,
                 dropout=dropout,
                 output_feed_forward_hidden_dim=output_feed_forward_hidden_dim,
                 initializer_range=initializer_range,
                 weighted_entity_threshold=weighted_entity_threshold,
                 include_null_embedding_in_dot_attention=include_null_embedding_in_dot_attention,
                 null_entity_id=null_entity_id)


    def get_metrics(self, reset: bool = False):
        metrics = super().get_metrics(reset)
        return metrics


    def unfreeze(self, mode):
        # don't hold an parameters directly, so do nothing
        self.disambiguator.unfreeze(mode)

    def forward(self,
                contextual_embeddings: torch.Tensor,
                tokens_mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                **kwargs):

        disambiguator_output = self.disambiguator(
            contextual_embeddings=contextual_embeddings,
            mask=tokens_mask,
            candidate_spans=candidate_spans,
            candidate_entities=candidate_entities['ids'],
            candidate_entity_priors=candidate_entity_priors,
            candidate_segment_ids=candidate_segment_ids,
            **kwargs
        )

        linking_scores = disambiguator_output['linking_scores']

        return_dict = disambiguator_output

        if 'gold_entities' in kwargs:
            loss_dict = self._compute_loss(
                    candidate_entities['ids'],
                    candidate_spans,
                    linking_scores,
                    kwargs['gold_entities']['ids']
            )
            return_dict.update(loss_dict)

        return return_dict


@Model.register("soldered_kg")
class SolderedKG(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 entity_linker: Model,
                 span_attention_config: Dict[str, int],
                 should_init_kg_to_bert_inverse: bool = True,
                 freeze: bool = False,
                 regularizer: RegularizerApplicator = None):
        super().__init__(vocab, regularizer)

        self.entity_linker = entity_linker
        self.entity_embedding_dim = self.entity_linker.disambiguator.entity_embedding_dim
        self.contextual_embedding_dim = self.entity_linker.disambiguator.contextual_embedding_dim

        self.weighted_entity_layer_norm = BertLayerNorm(self.entity_embedding_dim, eps=1e-5)
        init_bert_weights(self.weighted_entity_layer_norm, 0.02)

        self.dropout = torch.nn.Dropout(0.1)

        # the span attention layers
        assert len(span_attention_config) == 4
        config = BertConfig(
            0, # vocab size, not used
            hidden_size=span_attention_config['hidden_size'],
            num_hidden_layers=span_attention_config['num_hidden_layers'],
            num_attention_heads=span_attention_config['num_attention_heads'],
            intermediate_size=span_attention_config['intermediate_size']
        )
        self.span_attention_layer = SpanAttentionLayer(config)
        # already init inside span attention layer

        # for the output!
        self.output_layer_norm = BertLayerNorm(self.contextual_embedding_dim, eps=1e-5)

        self.kg_to_bert_projection = torch.nn.Linear(
                self.entity_embedding_dim, self.contextual_embedding_dim
        )

        self.should_init_kg_to_bert_inverse = should_init_kg_to_bert_inverse
        self._init_kg_to_bert_projection()

        self._freeze_all = freeze

    def _init_kg_to_bert_projection(self):
        if not self.should_init_kg_to_bert_inverse:
            return

        # the output projection we initialize from the bert to kg, after
        # we load the weights
        # projection as the pseudo-inverse
        # w = (entity_dim, contextual_embedding_dim)
        w = self.entity_linker.disambiguator.bert_to_kg_projector.weight.data.detach().numpy()
        w_pseudo_inv = np.dot(np.linalg.inv(np.dot(w.T, w)), w.T)
        b = self.entity_linker.disambiguator.bert_to_kg_projector.bias.data.detach().numpy()
        b_pseudo_inv = np.dot(w_pseudo_inv, b)
        self.kg_to_bert_projection.weight.data.copy_(torch.tensor(w_pseudo_inv))
        self.kg_to_bert_projection.bias.data.copy_(torch.tensor(b_pseudo_inv))

    def get_metrics(self, reset=False):
        return self.entity_linker.get_metrics(reset)

    def unfreeze(self, mode):
        if self._freeze_all:
            for p in self.parameters():
                p.requires_grad_(False)
            self.entity_linker.unfreeze('freeze')
            return

        if mode == 'entity_linking':
            # training the entity linker, fix parameters here
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for p in self.parameters():
                p.requires_grad_(True)

        # unfreeze will get called after loading weights in the case where
        # we pass a model archive to KnowBert, so re-init here
        self._init_kg_to_bert_projection()

        self.entity_linker.unfreeze(mode)

    def forward(self,
                contextual_embeddings: torch.Tensor,
                tokens_mask: torch.Tensor,
                candidate_spans: torch.Tensor,
                candidate_entities: torch.Tensor,
                candidate_entity_priors: torch.Tensor,
                candidate_segment_ids: torch.Tensor,
                **kwargs):

        linker_output = self.entity_linker(
                contextual_embeddings, tokens_mask,
                candidate_spans, candidate_entities, candidate_entity_priors,
                candidate_segment_ids, **kwargs)

        # update the span representations with the entity embeddings
        span_representations = linker_output['projected_span_representations']
        weighted_entity_embeddings = linker_output['weighted_entity_embeddings']
        spans_with_entities = self.weighted_entity_layer_norm(
                (span_representations +
                self.dropout(weighted_entity_embeddings)).contiguous()
        )

        # now run self attention between bert and spans_with_entities
        # to update bert.
        # this is done in projected dimension
        entity_mask = candidate_spans[:, :, 0] > -1
        span_attention_output = self.span_attention_layer(
                linker_output['projected_bert_representations'],
                spans_with_entities,
                entity_mask
        )
        projected_bert_representations_with_entities = span_attention_output['output']
        entity_attention_probs = span_attention_output["attention_probs"]

        # finally project back to full bert dimension!
        bert_representations_with_entities = self.kg_to_bert_projection(
                projected_bert_representations_with_entities
        )
        new_contextual_embeddings = self.output_layer_norm(
                (contextual_embeddings + self.dropout(bert_representations_with_entities)).contiguous()
        )

        return_dict = {'entity_attention_probs': entity_attention_probs,
                       'contextual_embeddings': new_contextual_embeddings,
                       'linking_scores': linker_output['linking_scores']}
        if 'loss' in linker_output:
            return_dict['loss'] = linker_output['loss']

        return return_dict


@Model.register("knowbert")
class KnowBert(BertPretrainedMetricsLoss):
    def __init__(self,
                 vocab: Vocabulary,
                 soldered_kgs: Dict[str, Model],
                 soldered_layers: Dict[str, int],
                 bert_model_name: str,
                 mode: str = None,
                 model_archive: str = None,
                 strict_load_archive: bool = True,
                 debug_cuda: bool = False,
                 remap_segment_embeddings: int = None,
                 regularizer: RegularizerApplicator = None):

        super().__init__(vocab, regularizer)

        self.remap_segment_embeddings = remap_segment_embeddings

        # get the LM + NSP parameters from BERT
        pretrained_bert = BertForPreTraining.from_pretrained(bert_model_name)
        self.pretrained_bert = pretrained_bert
        self.pretraining_heads = pretrained_bert.cls
        self.pooler = pretrained_bert.bert.pooler

        # the soldered kgs
        self.soldered_kgs = soldered_kgs
        for key, skg in soldered_kgs.items():
            self.add_module(key + "_soldered_kg", skg)

        # list of (layer_number, soldered key) sorted in ascending order
        self.layer_to_soldered_kg = sorted(
                [(layer, key) for key, layer in soldered_layers.items()]
        )
        # the last layer
        num_bert_layers = len(self.pretrained_bert.bert.encoder.layer)
        # the first element of the list is the index
        self.layer_to_soldered_kg.append([num_bert_layers - 1, None])

        if model_archive is not None:
            with tarfile.open(cached_path(model_archive), 'r:gz') as fin:
                # a file object
                weights_file = fin.extractfile('weights.th')
                state_dict = torch.load(weights_file, map_location=device_mapping(-1))
            self.load_state_dict(state_dict, strict=strict_load_archive)

        if remap_segment_embeddings is not None:
            # will redefine the segment embeddings
            new_embeddings = self._remap_embeddings(self.pretrained_bert.bert.embeddings.token_type_embeddings.weight)
            if new_embeddings is not None:
                del self.pretrained_bert.bert.embeddings.token_type_embeddings
                self.pretrained_bert.bert.embeddings.token_type_embeddings = new_embeddings

        assert mode in (None, 'entity_linking')
        self.mode = mode
        self.unfreeze()

        if debug_cuda:
            for m in self.modules():
                m.register_forward_hook(diagnose_forward_hook)
                m.register_backward_hook(diagnose_backward_hook)

    def _remap_embeddings(self, token_type_embeddings):
        embed_dim = token_type_embeddings.shape[1]
        if list(token_type_embeddings.shape) == [self.remap_segment_embeddings, embed_dim]:
            # already remapped!
            return None
        new_embeddings = torch.nn.Embedding(self.remap_segment_embeddings, embed_dim)
        new_embeddings.weight.data.copy_(token_type_embeddings.data[0, :])
        return new_embeddings


    def load_state_dict(self, state_dict, strict=True):
        if self.remap_segment_embeddings:
            # hack the embeddings!
            new_embeddings = self._remap_embeddings(state_dict['pretrained_bert.bert.embeddings.token_type_embeddings.weight'])
            if new_embeddings is not None:
                state_dict['pretrained_bert.bert.embeddings.token_type_embeddings.weight'] = new_embeddings.weight
        super().load_state_dict(state_dict, strict=strict)

    def unfreeze(self):
        if self.mode == 'entity_linking':
            # all parameters in BERT are fixed, just training the linker
            # linker specific params set below when calling soldered_kg.unfreeze
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for p in self.parameters():
                p.requires_grad_(True)

        for key in self.soldered_kgs.keys():
            module = getattr(self, key + "_soldered_kg")
            module.unfreeze(self.mode)

    def get_metrics(self, reset: bool = False):
        metrics = super().get_metrics(reset)

        for key in self.soldered_kgs.keys():
            module = getattr(self, key + "_soldered_kg")
            module_metrics = module.get_metrics(reset)
            for metric_key, val in module_metrics.items():
                metrics[key + '_' + metric_key] = val

        return metrics


    def forward(self, tokens=None, segment_ids=None, candidates=None,
                lm_label_ids=None, next_sentence_label=None, **kwargs):

        assert candidates.keys() == self.soldered_kgs.keys()

        mask = tokens['tokens'] > 0
        attention_mask = extend_attention_mask_for_bert(mask, get_dtype_for_module(self))
        contextual_embeddings = self.pretrained_bert.bert.embeddings(tokens['tokens'], segment_ids)

        output = {}
        start_layer_index = 0
        loss = 0.0

        gold_entities = kwargs.pop('gold_entities', None)


        for layer_num, soldered_kg_key in self.layer_to_soldered_kg:
            end_layer_index = layer_num + 1
            if end_layer_index > start_layer_index:
                # run bert from start to end layers
                for layer in self.pretrained_bert.bert.encoder.layer[
                                start_layer_index:end_layer_index]:
                    contextual_embeddings = layer(contextual_embeddings, attention_mask)
            start_layer_index = end_layer_index

            # run the SolderedKG component
            if soldered_kg_key is not None:
                soldered_kg = getattr(self, soldered_kg_key + "_soldered_kg")
                soldered_kwargs = candidates[soldered_kg_key]
                soldered_kwargs.update(kwargs)
                if gold_entities is not None and soldered_kg_key in gold_entities:
                    soldered_kwargs['gold_entities'] = gold_entities[soldered_kg_key]
                kg_output = soldered_kg(
                        contextual_embeddings=contextual_embeddings,
                        tokens_mask=mask,
                        **soldered_kwargs)

                if 'loss' in kg_output:
                    loss = loss + kg_output['loss']

                contextual_embeddings = kg_output['contextual_embeddings']
                output[soldered_kg_key] = {}
                for key in kg_output.keys():
                    if key != 'loss' and key != 'contextual_embeddings':
                        output[soldered_kg_key][key] = kg_output[key]

        # get the pooled CLS output
        pooled_output = self.pooler(contextual_embeddings)

        if lm_label_ids is not None or next_sentence_label is not None:
            # compute loss !
            masked_lm_loss, next_sentence_loss = self._compute_loss(
                    contextual_embeddings,
                    pooled_output,
                    lm_label_ids,
                    next_sentence_label)

            loss = loss + masked_lm_loss + next_sentence_loss

        if 'mask_indicator' in kwargs:
            self._compute_mrr(contextual_embeddings,
                              pooled_output,
                              lm_label_ids['lm_labels'],
                              kwargs['mask_indicator'])

        output['loss'] = loss
        output['pooled_output'] = pooled_output
        output['contextual_embeddings'] = contextual_embeddings

        return output

