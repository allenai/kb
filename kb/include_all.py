
from kb.kg_embedding import KGTupleReader, KGTupleModel
from kb.entity_linking import TokenCharactersIndexerTokenizer
from kb.entity_linking import CrossSentenceLinking
from kb.wordnet import WordNetFineGrainedSenseDisambiguationReader
from kb.wordnet import WordNetAllEmbedding
from kb.multitask import MultitaskDatasetReader, MultiTaskDataIterator
from kb.bert_pretraining_reader import BertPreTrainingReader
from kb.bert_tokenizer_and_candidate_generator import BertTokenizerAndCandidateGenerator, TokenizerAndCandidateGenerator
from kb.self_attn_bucket_iterator import SelfAttnBucketIterator
from kb.knowbert import KnowBert, BertPretrainedMaskedLM
from kb.bert_utils import GeLu
from kb.wiki_linking_reader import LinkingReader
from kb.kg_probe_reader import KgProbeReader

from kb.evaluation.classification_model import SimpleClassifier
from kb.evaluation.tacred_dataset_reader import TacredDatasetReader
from kb.evaluation.wic_dataset_reader import WicDatasetReader
from kb.evaluation.semeval2010_task8 import SemEval2010Task8Reader, SemEval2010Task8Metric
from kb.evaluation.fbeta_measure import FBetaMeasure
from kb.evaluation.ultra_fine_reader import UltraFineReader

from kb.common import F1Metric

from allennlp.models.archival import load_archive
from allennlp.models import Model

import json


@Model.register("from_archive")
class ModelArchiveFromParams(Model):
    """
    Loads a model from an archive
    """
    @classmethod
    def from_params(cls, vocab=None, params=None):
        """
        {"type": "from_archive", "archive_file": "path to archive",
         "overrides:" .... }

        "overrides" omits the "model" key
        """
        archive_file = params.pop("archive_file")
        overrides = params.pop("overrides", None)
        params.assert_empty("ModelArchiveFromParams")
        if overrides is not None:
            archive = load_archive(archive_file, overrides=json.dumps({'model': overrides.as_dict()}))
        else:
            archive = load_archive(archive_file)
        return archive.model

