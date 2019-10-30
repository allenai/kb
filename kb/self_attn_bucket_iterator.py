
import logging
import random
from collections import deque
from typing import List, Tuple, Iterable, cast, Dict, Deque

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary

import bisect

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from allennlp.data.iterators.bucket_iterator import sort_by_padding


SCHEDULES = {
    "base-24gb-bs64_fp32": [
        [64, 115],
        [32, 220],
        [16, 380],
        [8, 512]
    ],
    "base-12gb-fp32": [
        [32, 90],
        [16, 170],
        [8, 300],
        [4, 400],
        [2, 512]
    ],
    "base-11gb-fp32": [
        [32, 80],
        [16, 150],
        [8, 270],
        [4, 370],
        [2, 512]
    ],
    "base-24gb-fp32": [
        [32, 140],
        [16, 280],
        [8, 400],
        [4, 512],
    ],
}


@DataIterator.register("self_attn_bucket")
class SelfAttnBucketIterator(DataIterator):
    """
    Like a bucket iterator, but with a quadratic relationship between
    sequence length and batch size instead of linear.

    Has a fixed schedule of batch size vs sequence length.
    """
    def __init__(self,
                 batch_size_schedule: str,
                 iterator: DataIterator):

        if isinstance(batch_size_schedule, str):
            schedule = SCHEDULES[batch_size_schedule]
        else:
            # user is providing a dict directly
            schedule = batch_size_schedule

        # set batch size to max value in schedule
        batch_size = schedule[0][0]

        super().__init__(
            batch_size=batch_size,
            instances_per_epoch=iterator._instances_per_epoch,
            max_instances_in_memory=iterator._max_instances_in_memory,
            cache_instances=iterator._cache_instances,
            track_epoch=iterator._track_epoch,
            maximum_samples_per_batch=iterator._maximum_samples_per_batch
        )

        self.iterator = iterator

        # process the schedule
        self._schedule_batch_sizes = [ele[0] for ele in schedule]
        self._schedule_lengths = [ele[1] for ele in schedule]

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab
        self.iterator.index_with(vocab)

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        for batch in self.iterator._create_batches(instances, shuffle):
            # split after shuffling so smaller batches are kept together
            batch_instances = batch.instances

            # split if needed
            batch_length = -1
            for instance in batch_instances:
                instance.index_fields(self.vocab)
                field_lengths = instance.get_padding_lengths()
                batch_length = max(batch_length, field_lengths['tokens']['num_tokens'])

            # get the required batch size
            index = bisect.bisect_left(self._schedule_lengths, batch_length)
            if index == len(self._schedule_lengths):
                # this batch exceeds the maximum allowed, just skip it
                continue
            batch_size = self._schedule_batch_sizes[index]
            start = 0
            while start < len(batch_instances):
                end = start + batch_size
                yield Batch(batch_instances[start:end])
                start = end

