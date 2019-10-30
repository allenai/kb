
from typing import Dict, List, Iterator

from overrides import overrides

from allennlp.data.fields.field import DataArray, Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.common.util import pad_sequence_to_length

SEPERATOR = '*'


class DictField(Field):
    """
    dict with values as fields
    """
    def __init__(self, field_dict: Dict[str, Field]) -> None:
        self.field_dict = field_dict

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for field in self.field_dict.values():
            field.count_vocab_items(counter)

    @overrides
    def index(self, vocab: Vocabulary):
        for field in self.field_dict.values():
            field.index(vocab)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths = {}
        for key, field in self.field_dict.items():
            for sub_key, val in field.get_padding_lengths().items():
                padding_lengths[key + SEPERATOR + sub_key] = val
        return padding_lengths

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        # padding_lengths is flattened from the nested structure -- unflatten
        pl = {}
        for full_key, val in padding_lengths.items():
            key, _, sub_key = full_key.partition(SEPERATOR)
            if key not in pl:
                pl[key] = {}
            pl[key][sub_key] = val

        ret = {}
        for key, field in self.field_dict.items():
            ret[key] = field.as_tensor(pl[key])

        return ret

    @overrides
    def empty_field(self):
        return DictField({key: field.empty_field() for key, field in self.field_dict.items()})

    @overrides
    def batch_tensors(self, tensor_list_of_dict):
        ret = {}
        for key, field in self.field_dict.items():
            ret[key] = field.batch_tensors([t[key] for t in tensor_list_of_dict])
        return ret

    def __str__(self) -> str:
        return ""
