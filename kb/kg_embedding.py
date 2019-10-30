import torch
import numpy as np

from typing import List, Set, Dict

from collections import OrderedDict

from allennlp.data import DatasetReader, Token, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import Field, TextField
from allennlp.data.instance import Instance
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.models import Model
from allennlp.common import Registrable


@DatasetReader.register("kg_tuple")
class KGTupleReader(DatasetReader):
    """
    A knowledge graph consists of tuples (e1, r, e2).
    These are written in a tab separated text file with one tuple per line:
        e1\tr\te2
    """
    def __init__(
            self,
            entity_indexer: Dict[str, TokenIndexer] = None,
            relation_indexer: Dict[str, TokenIndexer] = None,
            extra_files_for_gold_pairs: List[str] = None
    ):
        lazy = False
        super().__init__(lazy)
    
        self.entity_indexer = entity_indexer or {"entity": SingleIdTokenIndexer("entity")}
        self.relation_indexer = relation_indexer or {"relation": SingleIdTokenIndexer("relation")}

        self.extra_files_for_gold_pairs = extra_files_for_gold_pairs

    def _read(self, file_path: str):
        """
        To create training instances:
            (1) read through entire file, and for each (e1, r) make a set of
                all valid e2
            (2) create (e2, r_reverse) set of all e1 to add reverse relations.
            (3) create binary labels
        
        To create validation / testing instances, follow the same procedure
        EXCEPT use all available data (train, dev, test) to construct the
        set of true entities.
        """
        if self.extra_files_for_gold_pairs is not None:
            files = list(self.extra_files_for_gold_pairs)
            include_as_instance = [False] * len(files)
        else:
            files = []
            include_as_instance = []

        files.append(file_path)
        include_as_instance.append(True)

        # (entity, relation) -> set(all valid entity2)
        full_graph = {}
        instances = OrderedDict()

        for fname, should_create_instance in zip(files, include_as_instance):
            with open(fname, 'r') as fin:
                for line in fin:
                    e1, r, e2 = [ele.strip() for ele in line.strip().split('\t')]
                    if (e1, r) not in full_graph:
                        full_graph[(e1, r)] = set()
                    full_graph[(e1, r)].add(e2)

                    r_reverse = r + '_reverse'
                    if (e2, r_reverse) not in full_graph:
                        full_graph[(e2, r_reverse)] = set()
                    full_graph[(e2, r_reverse)].add(e1)

                    if should_create_instance:
                        if (e1, r) not in instances:
                            instances[(e1, r)] = e2
                        if (e2, r_reverse) not in instances:
                            instances[(e2, r_reverse)] = e1

        # now create instances
        for (e1, r), e2 in instances.items():
            yield self.tuple_to_instance(e1, r, full_graph[(e1, r)], e2)

    def tuple_to_instance(self, e1: str, r: str, all_e2: Set[str], e2: str):
        # idea: create "text fields" then index them
        fields = {
            "entity": TextField([Token(e1)], self.entity_indexer),
            "relation": TextField([Token(r)], self.relation_indexer),
            "entity2": TextField([Token(t) for t in all_e2], self.entity_indexer),
            "entity2_target": TextField([Token(e2)], self.entity_indexer)
        }
        return Instance(fields)

    def text_to_instance(self):
        raise NotImplementedError


class KGTuplePredictor(torch.nn.Module, Registrable):
    pass


@KGTuplePredictor.register("tucker")
class TuckER(KGTuplePredictor):

    """
    TuckER: Tensor Factorization for Knowledge Graph Completion
    https://arxiv.org/pdf/1901.09590.pdf

    Basic idea: link tensor can be decomposed into a smaller "core" tensor,
    and 3 factor matrices (where for kbc, the entity matrix is two of these factor matrices,
    as we are comparing the same entity items i.e pairs of them).

    The scoring function for the tucker decomposition is:

    score(e1, e2, r) = W *_1 e1 *_2 r *_e2

    where *_n is the the tensor product along the nth mode.

    """

    def __init__(self,
	       	     num_entities: int,
	  	         num_relations: int,
                 entity_dim: int,
                 relation_dim: int,
                 input_dropout: float = 0.2,
                 hidden_dropout1: float = 0.2,
                 hidden_dropout2: float = 0.3):
        super().__init__()

        self.entities = torch.nn.Embedding(num_entities, entity_dim, padding_idx=0)
        self.relations = torch.nn.Embedding(num_relations, relation_dim, padding_idx=0)
        weight_shape = (relation_dim, entity_dim, entity_dim)
        self.W = torch.nn.Parameter(torch.randn(weight_shape, dtype=torch.float))

        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(hidden_dropout2)

        self.bn0 = torch.nn.BatchNorm1d(entity_dim)
        self.bn1 = torch.nn.BatchNorm1d(entity_dim)

        self.num_entities = num_entities

        self.reset_parameters()

    def get_entity_embedding(self):
        return self.entities

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.entities.weight.data)
        torch.nn.init.xavier_normal_(self.relations.weight.data)
        torch.nn.init.uniform_(self.W.data, -1, 1)

    def get_output_dim(self):
        return self.num_entities

    def forward(self, e1, rel):
        e1_embedded = self.entities(e1)
        rel_embedded = self.relations(rel)

        entity_dim = e1_embedded.size(1)
        rel_dim = rel_embedded.size(1)

        x = self.bn0(e1_embedded)
        x = self.input_dropout(x)
        x = x.view(-1, 1, entity_dim)

        # Pre-multiply by relation matrix
        W_mat = torch.mm(rel_embedded, self.W.view(rel_dim, -1))
        W_mat = W_mat.view(-1, entity_dim, entity_dim)
        W_mat = self.hidden_dropout1(W_mat)

        # Now all entities with all relations at once
        x = torch.bmm(x, W_mat)
        x = x.view(-1, entity_dim)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)

        x = torch.mm(x, self.entities.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred



def get_labels_tensor_from_indices(
        batch_size, num_embeddings, entity_ids, dtype=torch.float,
        label_smoothing=None
    ):
    # create a tensor of 0-1 that is shape (batch_size, num_embeddings)
    # entity_ids = (batch_size, max_num_positive_entities), type long
    # it contains the list of 1 label indices in (0, num_embeddings-1)
    labels = entity_ids.new_zeros(batch_size, num_embeddings, dtype=dtype)
    labels.scatter_add_(1, entity_ids, torch.ones_like(entity_ids, dtype=dtype))

    # remove the masking
    labels[:, 0] = 0.0

    # label smoothing
    if label_smoothing is not None:
        total_labels = labels.mean(dim=1, keepdim=True)
        labels = (1.0 - label_smoothing) * labels + label_smoothing * total_labels

    return labels


@Model.register("kg_tuple")
class KGTupleModel(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            kg_tuple_predictor: KGTuplePredictor,
            label_smoothing: float = 0.1,
            regularizer: RegularizerApplicator = None
    ) -> None:

        super().__init__(vocab, regularizer)

        self.kg_tuple_predictor = kg_tuple_predictor
        self.loss = torch.nn.BCELoss()

        self.dtype = torch.float
        self.num_entities = self.kg_tuple_predictor.get_output_dim()
        self.label_smoothing = label_smoothing

        self.ranking_and_hits = RankingAndHitsMetric()

    def get_entity_embedding(self):
        return self.kg_tuple_predictor.get_entity_embedding()


    def forward(self, entity, relation, entity2, entity2_target):
        # entity['entity'] = (batch_size, 1) with e1 ids
        # relation['relation'] = (batch_size, 1) with relation ids
        # entity2['entity'] = (batch_size, max_correct_entities) with
        #   ids of correct entities
        # entity2_target['entity'] = (batch_size, ) of the target entity2

        # run the prediction
        # (batch_size, num_entities)
        predicted_e2 = self.kg_tuple_predictor(
            entity['entity'].flatten(),
            relation['relation'].flatten()
        )

        # create the array with 0-1 values with gold entity2
        batch_size = entity['entity'].shape[0]
        labels = get_labels_tensor_from_indices(
                batch_size, self.num_entities, entity2['entity'],
                dtype=self.dtype, label_smoothing=self.label_smoothing
    
        )

        loss = self.loss(predicted_e2, labels)

        # metrics!
        if not self.training:
            self.ranking_and_hits(
                predicted_e2, entity2['entity'], entity2_target['entity'].flatten()
            )

        return {'loss': loss, 'predicted_entity2': predicted_e2}

    def get_metrics(self, reset: bool = False):
        if not self.training:
            return self.ranking_and_hits.get_metric(reset)
        else:
            return {}


class RankingAndHitsMetric:
    def __init__(self, hits_to_collect: List[int] = [1, 10]):
        self.hits_to_collect = hits_to_collect
        self.reset()

    def reset(self):
        self.hits = []
        for k in range(len(self.hits_to_collect)):
            self.hits.append([])
        self.ranks = []

    def get_metric(self, reset: bool = False):
        metrics = {
            'hits_{}'.format(h): np.mean(self.hits[i])
            for i, h in enumerate(self.hits_to_collect)
        }
        metrics['mean_rank'] = np.mean(self.ranks)
        metrics['mean_reciprocal_rank'] = np.mean(1./np.array(self.ranks))

        if reset:
            self.reset()

        return metrics

    def __call__(self, predicted, all_entity2, entity2):
        # predicted = (batch_size, num_entities)
        # all_entity2 = (batch_size, max_number_positive_entities) with ids
        #       of all known positive entities in all splits
        # entity2 = (batch_size, ) of the target entity2 id
        batch_size = predicted.shape[0]

        predicted = predicted.cpu()
        all_entity2 = all_entity2.cpu()
        entity2 = entity2.cpu()

        # zero out the actual entities EXCEPT for the target e2 entity
        for i in range(batch_size):
            e2_idx = entity2[i].item()
            e2_p = predicted[i][e2_idx].item()
            predicted[i][all_entity2[i]] = 0
            predicted[i][e2_idx] = e2_p

        # sort and rank
        max_values, argsort = torch.sort(predicted, 1, descending=True)
        argsort = argsort.numpy()
        for i in range(batch_size):
            # find the rank of the target entities
            rank = np.where(argsort[i]==entity2[i].item())[0][0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            self.ranks.append(rank + 1)

            # this could be done more elegantly, but here you go
            for ii, hits_level in enumerate(self.hits_to_collect):
                if rank <= hits_level:
                    self.hits[ii].append(1.0)
                else:
                    self.hits[ii].append(0.0)
