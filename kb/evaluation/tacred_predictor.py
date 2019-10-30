from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from kb.evaluation.tacred_dataset_reader import LABEL_MAP


REVERSE_LABEL_MAP = {y: x for x, y in LABEL_MAP.items()}


@Predictor.register('tacred')
class TacredPredictor(Predictor):
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        return REVERSE_LABEL_MAP[outputs['predictions']] + '\n'

