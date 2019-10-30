
import json

import torch

from pytorch_pretrained_bert.modeling import \
        BertLayer, BertAttention, BertSelfAttention, BertSelfOutput, \
        BertOutput, BertIntermediate, BertEncoder, BertLayerNorm, BertConfig

from allennlp.common.registrable import Registrable
from allennlp.training.metrics.metric import Metric

import spacy
from spacy.tokens import Doc



class MentionGenerator(Registrable):
    pass


class EntityEmbedder(Registrable):
    pass


def get_empty_candidates():
    """
    The mention generators always return at least one candidate, but signal
    it with this special candidate
    """
    return {
        "candidate_spans": [[-1, -1]],
        "candidate_entities": [["@@PADDING@@"]],
        "candidate_entity_priors": [[1.0]]
    }


# from https://spacy.io/usage/linguistic-features#custom-tokenizer-example
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


@Metric.register("f1_set")
class F1Metric(Metric):
    """
    A generic set based F1 metric.
    Takes two lists of predicted and gold elements and computes F1.
    Only requirements are that the elements are hashable.
    """
    def __init__(self, filter_func=None):
        self.reset()
        if filter_func is None:
            filter_func = lambda x: True
        self.filter_func = filter_func

    def reset(self):
        self._true_positives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        if reset:
            self.reset()
        return precision, recall, f1_measure

    def __call__(self, predictions, gold_labels):
        """
        predictions = batch of predictions that can be compared
        gold labels = list of gold labels

        e.g.
            predictions = [
                 [('ORG', (0, 1)), ('PER', (5, 8))],
                 [('MISC', (9, 13))]
            ]
            gold_labels = [
                [('ORG', (0, 1))],
                []
            ]

        elements must be hashable
        """
        assert len(predictions) == len(gold_labels)

        for pred, gold in zip(predictions, gold_labels):
            s_gold = set(g for g in gold if self.filter_func(g))
            s_pred = set(p for p in pred if self.filter_func(p))

            for p in s_pred:
                if p in s_gold:
                    self._true_positives += 1
                else:
                    self._false_positives += 1

            for p in s_gold:
                if p not in s_pred:
                    self._false_negatives += 1


def get_dtype_for_module(module):
    # gets dtype for module parameters, for fp16 support when casting
    # we unfortunately can't set this during module construction as module
    # will be moved to GPU or cast to half after construction.
    return next(module.parameters()).dtype

def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad_(requires_grad)

def extend_attention_mask_for_bert(mask, dtype):
    # mask = (batch_size, timesteps)
    # returns an attention_mask useable with BERT
    # see: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L696
    extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def init_bert_weights(module, initializer_range, extra_modules_without_weights=()):
    # these modules don't have any weights, other then ones in submodules,
    # so don't have to worry about init
    modules_without_weights = (
        BertEncoder, torch.nn.ModuleList, torch.nn.Dropout, BertLayer,
        BertAttention, BertSelfAttention, BertSelfOutput,
        BertOutput, BertIntermediate
    ) + extra_modules_without_weights


    # modified from pytorch_pretrained_bert
    def _do_init(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            m.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(m, BertLayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        elif isinstance(m, modules_without_weights):
            pass
        else:
            raise ValueError(str(m))

        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

    for mm in module.modules():
        _do_init(mm)

def get_linear_layer_init_identity(dim):
    ret = torch.nn.Linear(dim, dim)
    ret.weight.data.copy_(torch.eye(dim))
    ret.bias.data.fill_(0.0)
    return ret



class JsonFile:
    '''
    A flat text file where each line is one json object

    # to read though a file line by line
    with JsonFile('file.json', 'r') as fin:
        for line in fin:
            # line is the deserialized json object
            pass


    # to write a file object by object
    with JsonFile('file.json', 'w') as fout:
        fout.write({'key1': 5, 'key2': 'token'})
        fout.write({'key1': 0, 'key2': 'the'})
    '''

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __iter__(self):
        for line in self._file:
            yield json.loads(line)

    def write(self, item):
        item_as_json = json.dumps(item, ensure_ascii=False)
        encoded = '{0}\n'.format(item_as_json)
        self._file.write(encoded)

    def __enter__(self):
        self._file = open(*self._args, **self._kwargs)
        self._file.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.__exit__(exc_type, exc_val, exc_tb)

