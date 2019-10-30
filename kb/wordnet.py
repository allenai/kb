"""
Notes on wordnet ids:
    in KG embeddings, have both synset and lemma nodes:
        synsets are keyed by something like able.a.01register("wordnet_mention_generator")
        each synset has a number of lemmas, keyed by something like able%3:00:00::

    In WSD task, you are given (lemma, pos) and asked to predict the lemma
        key, e.g. (able, adj) -> which synset do we get?

    Internally, we use the able.a.01 key for synsets, but maintain a map
    from (lemma, pos, internal key) -> external key for evaluation with semcor.
"""


import torch

import random

from typing import List, Dict

from allennlp.data import DatasetReader, Token, Vocabulary, Tokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, TokenCharactersIndexer
from allennlp.data.fields import Field, TextField, ListField, SpanField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.models import Model
from allennlp.common.file_utils import cached_path


from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from collections import defaultdict

from kb.common import JsonFile, get_empty_candidates
from kb.kg_embedding import KGTuplePredictor
from kb.entity_linking import EntityLinkingReader

from kb.common import WhitespaceTokenizer, MentionGenerator, init_bert_weights, EntityEmbedder

from pytorch_pretrained_bert.modeling import BertLayerNorm


import numpy as np
import h5py

import spacy
from spacy.tokens import Doc

class WordNetSpacyPreprocessor:
    """
    A "preprocessor" that really does POS tagging and lemmatization using spacy,
    plus some hand crafted rules.

    allennlp tokenizers take strings and return lists of Token classes.
    we'll run spacy first, then modify the POS / lemmas as needed, then
    return a new list of Token
    """
    def __init__(self, whitespace_tokenize_only: bool = False):
        self.nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'textcat'])
        if whitespace_tokenize_only:
            self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)

        # spacy POS are similar, but not exactly the same as wordnet,
        # so need this conversion for tags that need to be mapped
        self.spacy_to_wordnet_map = {
            'PROPN': 'NOUN'
        }

    def __call__(self, text: str) -> List[Token]:
        spacy_doc = self.nlp(text)

        # create allennlp tokens
        normalized_tokens = [
            Token(spacy_token.text,
                  pos_=self.spacy_to_wordnet_map.get(spacy_token.pos_, spacy_token.pos_),
                  lemma_=spacy_token.lemma_
            )

            for spacy_token in spacy_doc
            if not spacy_token.is_space
        ]

        return normalized_tokens


def _norm_lemma(lemma_str):
    return lemma_str.replace('_', ' ').replace('-', ' ')


WORDNET_TO_SEMCOR_POS_MAP = {
    'n': 'NOUN',  # %1
    'v': 'VERB',  # %2
    'a': 'ADJ',   # %3
    'r': 'ADV',   # %4
    's': 'ADJ',   # %5
}


def load_candidate_maps(fname, topk=30, count_smoothing=1):
    """
    Load the candidate maps from the entity file.

    entity_file is the jsonl dump from extract_wordnet.py

    returns:
        candidates[Dict[normalized lemma string] -> candidates
        lemma_id_to_synset_id = Dict["able%3:00:00"] -> "able.a.01"

    each value candidates list is:
        [candidate1_metadata, candidate2_metadata, etc]
    where candidate_metadata is a dict with keys:
            synset_id, lemma_id, pos (n, v, a,   ), prior

    The lemmas are underscore and hyphen normalized for training.

    topk = keep this many of the top candidates for each lemma
    count_smoothing = use this for smoothing
        if count_smoothing < 0 then don't normalize lemmas, just return raw counts
    """
    def _update(d, key, m):
        if key not in d:
            d[key] = []
        d[key].append(m)

    def _trim_and_normalize(d, num, smoothing):
        for key in d:
            all_candidates = d[key]
            if len(all_candidates) > num:
                # sort by count and trim
                # sorted sorts ascending by default, we want decending by count
                sorted_candidates = sorted(all_candidates, key=lambda x: x['prior'], reverse=True)
                trimmed_candidates = sorted_candidates[:num]
            else:
                trimmed_candidates = all_candidates

            if smoothing >= 0:
                sum_count = sum(ele['prior'] + smoothing for ele in trimmed_candidates)
                for cand in trimmed_candidates:
                    cand['prior'] = (cand['prior'] + smoothing) / sum_count
            d[key] = trimmed_candidates

    candidates = {}
    lemma_id_to_synset_id = {}

    with JsonFile(cached_path(fname), 'r') as fin:
        for entity in fin:
            if entity['type'] == 'lemma':
                lemma_id = entity['id']
                lemma_str = lemma_id.partition('%')[0]
                synset_id = entity['synset']

                metadata = {'synset_id': synset_id,
                            'lemma_id': lemma_id,
                            'pos': entity['pos'],
                            'prior': entity['count']}

                # normalize the lemma_str
                lemma_str_normalized = _norm_lemma(lemma_str)
                _update(candidates, lemma_str_normalized, metadata)

                lemma_id_to_synset_id[lemma_id] = synset_id


    # now trim to top k and normalize the prior
    _trim_and_normalize(candidates, topk, count_smoothing)

    return candidates, lemma_id_to_synset_id


# Unsupervised setting for LM:
#   raw data -> use spacy to get lemma -> look up all candidates normalizing
#       - and _
#
# With annotated data:
#       at train time:
#           given gold spans and entity ids:
#               map semcor tokens to flat token sequence + gold ids + gold spans
#               look up all candidate spans using raw data approach ignoring POS and lemma
#               remove generic entity types
#               restrict candidate spans to just those that have annotated senses
#               compute the recall of gold span / entity from pruned candidate lsit (for MWE separate from single words)
#
#       at test time:
#           given gold POS and lemma, get candidates.
#           for generic entity types, use heuristic to restrict candidates
#           should have near 100% recall of gold span
#           and first sense baseline should be high

def _update_candidate_list(c, s, e, p):
    c['candidate_spans'].append(s)
    c['candidate_entities'].append(e)
    c['candidate_entity_priors'].append(p)

@MentionGenerator.register("wordnet_mention_generator")
class WordNetCandidateMentionGenerator(MentionGenerator):
    """
    Generate lists of candidate entities. Provides several methods that
    process input text of various format to produce mentions.

    Each text is represented by:
            {'tokenized_text': List[str],
             'candidate_spans': List[List[int]] list of (start, end) indices for candidates,
                    where span is tokenized_text[start:(end + 1)]
             'candidate_entities': List[List[str]] = for each entity,
                    the candidates to link to. value is synset id, e.g
                    able.a.02 or hot_dog.n.01
             'candidate_entity_priors': List[List[float]]
        }
    """
    def __init__(
            self,
            entity_file: str,
            max_entity_length: int = 7,
            max_number_candidates: int = 30,
            count_smoothing: int = 1,
            use_surface_form: bool = False,
            random_candidates: bool = False):

        self._raw_data_processor = WordNetSpacyPreprocessor()
        self._raw_data_processor_whitespace = WordNetSpacyPreprocessor(
                whitespace_tokenize_only=True
        )

        self._candidate_list, self._lemma_to_synset = load_candidate_maps(
                entity_file, count_smoothing=-1
        )
        # candidate_list[hog dog] -> [all candidate lemmas]

        self._entity_synsets = {
                #'location%1:03:00::': 'location.n.01',  # (LOC)
                #'person%1:03:00::': 'person.n.01',    # (PER)
                #'group%1:03:00::': 'group.n.01'      # (ORG)
                'location': 'location.n.01',  # (LOC)
                'person': 'person.n.01',    # (PER)
                'group': 'group.n.01'      # (ORG)
        }
        self._entity_lemmas = {
                'location%1:03:00::',
                'person%1:03:00::',
                'group%1:03:00::',
        }

        self._max_entity_length = max_entity_length
        self._max_number_candidates = max_number_candidates
        self._count_smoothing = count_smoothing
        self._use_surface_form = use_surface_form

        self._random_candidates = random_candidates
        if self._random_candidates:
            self._unique_synsets = list(set(self._lemma_to_synset.values()))

    def get_mentions_with_gold_spans(
            self, gold_annotations
    ):
        """
        use for training with semcor -- it will use the full unrestricted
            generator, but restrict to just the gold annotation spans, without
            using the gold lemma or POS.
            
        remove generic entity types (PER, LOC, ORG)
        restrict candidate spans to just those that have annotated senses
        """
        tokenized_text = gold_annotations['tokenized_text']

        text = ' '.join(gold_annotations['tokenized_text'])
        candidates = self.get_mentions_raw_text(text, whitespace_tokenize=True,
            allow_empty_candidates=True)

        # each gold annotation needs to be in output
        # will look up candidates by (start, end) indices so remap candidates
        candidates_by_endpoints = {
            tuple(start_end): {'entities': ent, 'priors': pri}
            for start_end, ent, pri in zip(
                    candidates['candidate_spans'],
                    candidates['candidate_entities'],
                    candidates['candidate_entity_priors']
            )
        }

        filtered_candidates = {
                      'tokenized_text': tokenized_text,
                      'candidate_spans': [],
                      'candidate_entities': [],
                      'candidate_entity_priors': []
        }

        for k in range(len(gold_annotations['gold_spans'])):
            span = gold_annotations['gold_spans'][k]
            lemma = gold_annotations['gold_lemmas'][k]
            pos = gold_annotations['gold_pos'][k]
            lemma_id = gold_annotations['gold_lemma_ids'][k]

            if lemma_id in self._entity_lemmas:
                # skip
                continue

            span_candidates = candidates_by_endpoints.get(tuple(span))
            if span_candidates is None:
                # this one wasn't found by candidate generation
                continue

            # add to the list
            _update_candidate_list(filtered_candidates, span,
                                   span_candidates['entities'],
                                   span_candidates['priors'])

        return filtered_candidates

    def get_mentions_from_gold_span_lemma_pos(
            self, gold_annotations
    ):
        """
        apply heuristic for generic entity types
        restrict candidate spans to just those that have
            annotated senses
        use the gold lemma and POS to further restrict entity types

        gold_annotations is output from unpack_wsd_training_instance, has keys
            {'tokenized_text': tokenized_text,
            'gold_spans': gold_spans,
            'gold_entities': gold_entities,
            'gold_lemmas': gold_lemmas,
            'gold_pos': gold_pos,
            'gold_ids': gold_ids}
        """
        # each gold annotation needs to be in output
        # need one output for each gold span
        tokenized_text = gold_annotations['tokenized_text']
        candidates = {'tokenized_text': tokenized_text,
                      'candidate_spans': [],
                      'candidate_entities': [],
                      'candidate_entity_priors': []}

        tokenized_text = gold_annotations['tokenized_text']
        for k in range(len(gold_annotations['gold_spans'])):
            span = gold_annotations['gold_spans'][k]
            lemma = gold_annotations['gold_lemmas'][k]
            pos = gold_annotations['gold_pos'][k]

            # check for named entities
            if pos == 'NOUN' and lemma in self._entity_synsets:
                if lemma != tokenized_text[span[0]]:
                    # hack, assume it's the generic entity type
                    candidate_ids = [self._entity_synsets[lemma]]
                    candidate_priors = [1.0]
                    _update_candidate_list(
                            candidates, span, candidate_ids, candidate_priors
                    )
                    continue

            # look up by (lemma, pos)
            normalized_lemma = _norm_lemma(lemma)
            all_candidates = self._candidate_list[normalized_lemma]
            # restrict to pos
            cand_entities = []
            cand_priors = []
            for cand in all_candidates:
                if WORDNET_TO_SEMCOR_POS_MAP[cand['pos']] == pos:
                    cand_entities.append(cand['synset_id'])
                    cand_priors.append(cand['prior'])
            # renormalize prior
            sum_prior = sum(cand_priors) + len(cand_priors) * self._count_smoothing
            norm_prior = [(p + self._count_smoothing) / sum_prior for p in cand_priors]
            _update_candidate_list(candidates, span, cand_entities, norm_prior)

        return candidates

    def get_mentions_raw_text(
                self,
                text: str,
                whitespace_tokenize: bool = False,
                allow_empty_candidates: bool = False,
        ):
        """
        returns:
            {'tokenized_text': List[str],
             'candidate_spans': List[List[int]] list of (start, end) indices for candidates,
                    where span is tokenized_text[start:(end + 1)]
             'candidate_entities': List[List[str]] = for each entity,
                    the candidates to link to. value is synset id, e.g
                    able.a.02 or hot_dog.n.01
             'candidate_entity_priors': List[List[float]]
        }
        """
        if whitespace_tokenize:
            tokenized = self._raw_data_processor_whitespace(text)
        else:
            tokenized = self._raw_data_processor(text)

        tokenized_text = [token.text for token in tokenized]

        # will look up by both lemma (and the tokenized text if use_surface_form)
        # lowercase and remove '.'
        lemmas = [token.lemma_.lower().replace('.', '') for token in tokenized]
        clist = [lemmas]
        if self._use_surface_form:
            normed_tokens = [
                token.lower().replace('.', '') for token in tokenized_text
            ]
            clist.append(normed_tokens)
            # remove ones that match the lemma
           # filtered_tokens = [None if t == l else t for l, t in  zip(lemmas, normed_tokens)]
            #clist.append(filtered_tokens)

        # look for candidates
        # 1. create lemma string key
        # 2. look up candidates
        # 3. combine candidates from lemmas and tokens
        # 4. sort and normalize the candidates

        # keep track of the candidates hashed by (start, end) indices
        # (start, end) -> [list of candidate dicts that will be sorted / normalized]
        candidates_by_span = defaultdict(lambda: list())
        n = len(tokenized_text)
        for start in range(n):
            for end in range(start, min(n, start + self._max_entity_length - 1)):
                for ci, cc in enumerate(clist):
                    # only consider strings that don't begin/end with '-'
                    # and surface forms that are different from lemmas
                    if cc[start] != '-' and cc[end] != '-' \
                        and (ci == 0 or cc[start:(end+1)] != lemmas[start:(end+1)]):
                        candidate_lemma = ' '.join([t for t in cc[start:(end+1)] if t != '-'])
                        if candidate_lemma in self._candidate_list:
                            candidate_metadata = self._candidate_list[candidate_lemma]
                            span_key = (start, end)
                            candidates_by_span[span_key].extend(candidate_metadata)

        # trim and normalize the candidates
        candidate_spans = []
        candidate_entities = []
        candidate_entity_priors = []
        for span_key, s_candidates in candidates_by_span.items():
            if len(s_candidates) > self._max_number_candidates:
                # sort by count and trim
                # sorted sorts ascending by default, we want decending by count
                sorted_candidates = sorted(s_candidates, key=lambda x: x['prior'], reverse=True)
                trimmed_candidates = sorted_candidates[:self._max_number_candidates]
            else:
                trimmed_candidates = s_candidates

            # normalize the counts
            sum_count = sum([ele['prior'] + self._count_smoothing for ele in trimmed_candidates])
            candidate_spans.append([span_key[0], span_key[1]])
            candidate_entities.append([c['synset_id'] for c in trimmed_candidates])
            candidate_entity_priors.append(
                [(c['prior'] + self._count_smoothing) / sum_count
                for c in trimmed_candidates]
            )

        if self._random_candidates:
            # randomly replace the candidate_entities
            for i in range(len(candidate_entities)):
                rand_candidates = list(candidate_entities[i])
                for j in range(len(rand_candidates)):
                    rand_candidate = random.choice(self._unique_synsets)
                    rand_candidates[j] = rand_candidate
                candidate_entities[i] = rand_candidates

        ret = {'tokenized_text': tokenized_text,
               'candidate_spans': candidate_spans,
               'candidate_entities': candidate_entities,
               'candidate_entity_priors': candidate_entity_priors}

        if not allow_empty_candidates and len(candidate_spans) == 0:
            # no candidates found, substitute the padding entity id
            ret.update(get_empty_candidates())

        return ret


def unpack_wsd_training_instance(context):
    """
    context is a list of tokens from semcor or the WSD evaluation datasets.
    each token is a dict with keys 'token', 'senses', etc.
    Some of the tokens are multi-word expressions.

    Returns:
        tokenized_text: a list of tokenized text where multi-word expressions
            are flattened
        gold_spans: a list of (start, end) indices of the annotated senses
            these are indexed such that tokenized_text[start:(end+1)] gives the
            span
        gold_lemma_ids: a list the gold entity strings where
            each is given as the lemma + sense, e.g. able%3:01:05::
        gold_lemmas: a list of the gold lemmas without normalization, e.g.
            hot_dog
        gold_pos: a list of the POS tags as read from file, e.g.

            NOUN, VERB, ADJ, ADV, with ADJ == 5:...
        gold_ids: a list of the gold ids as read from file, e.g.
            "d351.s073.t001" or "example_definition.515"
    """
    tokenized_text = []
    gold_spans = []
    gold_lemma_ids = []
    gold_lemmas = []
    gold_pos = []
    gold_ids = []

    start = 0
    for token in context:
        this_tokens = token['token'].replace('-', ' - ').split()
        tokenized_text.extend(this_tokens)

        n = len(this_tokens)
        end = start + n - 1

        if 'senses' in token:
            gold_spans.append([start, end])
            lemma = token['lemma']
            gold_lemmas.append(lemma)
            # just get the first
            gold_lemma_ids.append(lemma + '%' + token['senses'][0])
            gold_pos.append(token['pos'])
            gold_ids.append(token['id'])

        start = end + 1

    return {'tokenized_text': tokenized_text,
            'gold_spans': gold_spans,
            'gold_lemma_ids': gold_lemma_ids,
            'gold_lemmas': gold_lemmas,
            'gold_pos': gold_pos,
            'gold_ids': gold_ids}


@DatasetReader.register("wordnet_fine_grained")
class WordNetFineGrainedSenseDisambiguationReader(EntityLinkingReader):
    """
    Dataset reader for WSD annotated data.

    At train time, will use predicted pos and lemma, but gold spans as the
        data is not exhaustively annotated.
    At test time, use gold lemma and POS tags, consistent with existing
        task definitions.
    """

    def __init__(self,
                 wordnet_entity_file: str,
                 token_indexers: Dict[str, TokenIndexer],
                 entity_indexer: TokenIndexer,
                 is_training: bool,
                 use_surface_form: bool = False,
                 should_remap_span_indices: bool = True,
                 extra_candidate_generators: Dict[str, MentionGenerator] = None):

        super().__init__(False)

        self.mention_generator = WordNetCandidateMentionGenerator(
                wordnet_entity_file, use_surface_form=use_surface_form
        )

        self.token_indexers = token_indexers
        self.entity_indexer = {"ids": entity_indexer}
        self.is_training = is_training
        self.should_remap_span_indices = should_remap_span_indices

        self.extra_candidate_generators = extra_candidate_generators

    def _read(self, file_path: str):
        with JsonFile(cached_path(file_path), 'r') as fin:
            for sentence in fin:
                gold_annotations = unpack_wsd_training_instance(sentence)
                gold_span_to_entity_id = {
                    tuple(gs): self.mention_generator._lemma_to_synset[gi]
                    for gs, gi in zip(
                        gold_annotations['gold_spans'],
                        gold_annotations['gold_lemma_ids']
                    )
                }
                gold_span_to_data_id = {
                    tuple(gs): gid
                    for gs, gid in zip(
                        gold_annotations['gold_spans'],
                        gold_annotations['gold_ids']
                    )
                }

                # get the candidates
                if self.is_training:
                    candidates = self.mention_generator.get_mentions_with_gold_spans(gold_annotations)
                else:
                    candidates = self.mention_generator.get_mentions_from_gold_span_lemma_pos(gold_annotations)

                # map the original gold lemma_id to the synset_id
                gold_entities = [

                    # value is synset_id
                    gold_span_to_entity_id[tuple(candidate_span)]

                    for candidate_span in candidates['candidate_spans']

                ]

                gold_data_ids = [
                    gold_span_to_data_id[tuple(candidate_span)]
                    for candidate_span in candidates['candidate_spans']
                ]

                if len(candidates['candidate_spans']) > 0:
                    yield self.text_to_instance(
                            gold_annotations['tokenized_text'],
                            candidates['candidate_entities'],
                            candidates['candidate_spans'],
                            candidates['candidate_entity_priors'],
                            gold_entities,
                            gold_data_ids
                    )


    def text_to_instance(self,
                         tokens: List[str],
                         candidate_entities: List[List[str]],
                         candidate_spans: List[List[int]],
                         candidate_entity_prior: List[List[float]],
                         gold_entities: List[str] = None,
                         gold_data_ids: List[str] = None):

        # prior needs to be 2D and full
        # can look like [[0.2, 0.8], [1.0]]  if one candidate for second
        # candidate span and two candidates for first
        max_cands = max(len(p) for p in candidate_entity_prior)
        for p in candidate_entity_prior:
            if len(p) < max_cands:
                p.extend([0.0] * (max_cands - len(p)))
        np_prior = np.array(candidate_entity_prior)

        fields = {
            "tokens": TextField([Token(t) for t in tokens],
                      token_indexers=self.token_indexers),

            # join by space, then retokenize in the "character indexer"
            "candidate_entities": TextField(
                [Token(" ".join(candidate_list)) for candidate_list in candidate_entities],
                token_indexers=self.entity_indexer),
            "candidate_entity_prior": ArrayField(np.array(np_prior)),
            # only one sentence
            "candidate_segment_ids": ArrayField(
                np.array([0] * len(candidate_entities)), dtype=np.int
            )
        }

        if gold_entities is not None:
            fields["gold_entities"] =  TextField([Token(entity) for entity in gold_entities],
                                                  token_indexers=self.entity_indexer)
        if gold_data_ids is not None:
            fields["gold_data_ids"] = MetadataField(gold_data_ids)

        span_fields = []
        for span in candidate_spans:
            span_fields.append(SpanField(span[0], span[1], fields['tokens']))
        fields['candidate_spans'] = ListField(span_fields)

        if self.extra_candidate_generators:
            tokens = " ".join(tokens)
            extra_candidates = {
                    key: generator.get_mentions_raw_text(tokens, whitespace_tokenize=True)
                    for key, generator in self.extra_candidate_generators.items()
            }
            fields['extra_candidates'] = MetadataField(extra_candidates)

        return Instance(fields, should_remap_span_indices=self.should_remap_span_indices)


@EntityEmbedder.register('wordnet_all_embeddings')
class WordNetAllEmbedding(torch.nn.Module, EntityEmbedder):
    """
    Combines pretrained fixed embeddings with learned POS embeddings.

    Given entity candidate list:
        - get list of unique entity ids
        - look up
        - concat POS embedding
        - linear project
        - remap to candidate embedding shape
    """
    POS_MAP = {
        '@@PADDING@@': 0,
        'n': 1,
        'v': 2,
        'a': 3,
        'r': 4,
        's': 5,
        # have special POS embeddings for mask / null, so model can learn
        # it's own representation for them
        '@@MASK@@': 6,
        '@@NULL@@': 7,
        '@@UNKNOWN@@': 8
    }

    def __init__(self,
                 embedding_file: str,
                 entity_dim: int,
                 entity_file: str = None,
                 vocab_file: str = None,
                 entity_h5_key: str = 'conve_tucker_infersent_bert',
                 dropout: float = 0.1,
                 pos_embedding_dim: int = 25,
                 include_null_embedding: bool = False):
        """
        pass pos_emedding_dim = None to skip POS embeddings and all the
            entity stuff, using this as a pretrained embedding file
            with feedforward
        """

        super().__init__()

        if pos_embedding_dim is not None:
            # entity_id -> pos abbreviation, e.g.
            # 'cat.n.01' -> 'n'
            # includes special, e.g. '@@PADDING@@' -> '@@PADDING@@'
            entity_to_pos = {}
            with JsonFile(cached_path(entity_file), 'r') as fin:
                for node in fin:
                    if node['type'] == 'synset':
                        entity_to_pos[node['id']] = node['pos']
            for special in ['@@PADDING@@', '@@MASK@@', '@@NULL@@', '@@UNKNOWN@@']:
                entity_to_pos[special] = special
    
            # list of entity ids
            entities = ['@@PADDING@@']
            with open(cached_path(vocab_file), 'r') as fin:
                for line in fin:
                    entities.append(line.strip())
    
            # the map from entity index id -> pos embedding id,
            # will use for POS embedding lookup
            entity_id_to_pos_index = [
                 self.POS_MAP[entity_to_pos[ent]] for ent in entities
            ]
            self.register_buffer('entity_id_to_pos_index', torch.tensor(entity_id_to_pos_index))
    
            self.pos_embeddings = torch.nn.Embedding(len(entities), pos_embedding_dim)
            init_bert_weights(self.pos_embeddings, 0.02)

            self.use_pos = True
        else:
            self.use_pos = False

        # load the embeddings
        with h5py.File(cached_path(embedding_file), 'r') as fin:
            entity_embeddings = fin[entity_h5_key][...]
        self.entity_embeddings = torch.nn.Embedding(
                entity_embeddings.shape[0], entity_embeddings.shape[1],
                padding_idx=0
        )
        self.entity_embeddings.weight.data.copy_(torch.tensor(entity_embeddings).contiguous())

        if pos_embedding_dim is not None:
            assert entity_embeddings.shape[0] == len(entities)
            concat_dim = entity_embeddings.shape[1] + pos_embedding_dim
        else:
            concat_dim = entity_embeddings.shape[1]

        self.proj_feed_forward = torch.nn.Linear(concat_dim, entity_dim)
        init_bert_weights(self.proj_feed_forward, 0.02)

        self.dropout = torch.nn.Dropout(dropout)

        self.entity_dim = entity_dim

        self.include_null_embedding = include_null_embedding
        if include_null_embedding:
            # a special embedding for null
            entities = ['@@PADDING@@']
            with open(cached_path(vocab_file), 'r') as fin:
                for line in fin:
                    entities.append(line.strip())
            self.null_id = entities.index("@@NULL@@")
            self.null_embedding = torch.nn.Parameter(torch.zeros(entity_dim))
            self.null_embedding.data.normal_(mean=0.0, std=0.02)

    def get_output_dim(self):
        return self.entity_dim

    def get_null_embedding(self):
        return self.null_embedding

    def forward(self, entity_ids):
        """
        entity_ids = (batch_size, num_candidates, num_entities) array of entity
            ids

        returns (batch_size, num_candidates, num_entities, embed_dim)
            with entity embeddings
        """
        # get list of unique entity ids
        unique_ids, unique_ids_to_entity_ids = torch.unique(entity_ids, return_inverse=True)
        # unique_ids[unique_ids_to_entity_ids].reshape(entity_ids.shape)

        # look up (num_unique_embeddings, full_entity_dim)
        unique_entity_embeddings = self.entity_embeddings(unique_ids.contiguous()).contiguous()

        # get POS tags from entity ids (form entity id -> pos id embedding)
        # (num_unique_embeddings)
        if self.use_pos:
            unique_pos_ids = torch.nn.functional.embedding(unique_ids, self.entity_id_to_pos_index).contiguous()
            # (num_unique_embeddings, pos_dim)
            unique_pos_embeddings = self.pos_embeddings(unique_pos_ids).contiguous()
            # concat
            entity_and_pos = torch.cat([unique_entity_embeddings, unique_pos_embeddings], dim=-1)
        else:
            entity_and_pos = unique_entity_embeddings

        # run the ff
        # (num_embeddings, entity_dim)
        projected_entity_and_pos = self.dropout(self.proj_feed_forward(entity_and_pos.contiguous()))

        # replace null if needed
        if self.include_null_embedding:
            null_mask = unique_ids == self.null_id
            projected_entity_and_pos[null_mask] = self.null_embedding

        # remap to candidate embedding shape
        return projected_entity_and_pos[unique_ids_to_entity_ids].contiguous()
