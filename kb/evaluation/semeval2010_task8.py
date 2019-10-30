
from typing import Dict
from typing import Iterable

import tempfile
import subprocess
import re

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, MetadataField
from allennlp.data.instance import Instance
from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator
from kb.common import JsonFile

from allennlp.common.file_utils import cached_path
from allennlp.training.metrics import Metric


def to_jsonl(input_file, outprefix, train_or_test):
    import random

    with open(input_file, 'r') as fin:
        lines = fin.readlines()

    examples = []
    k = 0
    while k < len(lines):
        line = lines[k]
        ls = line.split('\t')
        try:
            sent_id = int(ls[0])
        except ValueError:
            sent_id = None

        if sent_id is not None and sent_id >= 1 and sent_id <= 10717:
            # a line with a train or test example
            # remove the " "
            sentence = ls[1].strip()[1:-1]

            # get the label
            k += 1
            label = lines[k].strip()

            example = {'sentence': sentence, 'label': label, 'sent_id': sent_id}
            examples.append(example)

        # go to the next line
        k += 1

    if train_or_test == 'train':
        # split into train / dev
        random.shuffle(examples)

        with JsonFile(outprefix + '/train.json', 'w') as fout:
            for example in examples[:7500]:
                fout.write(example)

        with JsonFile(outprefix + '/dev.json', 'w') as fout:
            for example in examples[7500:]:
                fout.write(example)
    else:
        with JsonFile(outprefix + '/test.json', 'w') as fout:
            for example in examples:
                fout.write(example)


LABEL_MAP = {
 'Other': 0,
 'Cause-Effect(e1,e2)': 1,
 'Cause-Effect(e2,e1)': 2,
 'Component-Whole(e1,e2)': 3,
 'Component-Whole(e2,e1)': 4,
 'Content-Container(e1,e2)': 5,
 'Content-Container(e2,e1)': 6,
 'Entity-Destination(e1,e2)': 7,
 'Entity-Destination(e2,e1)': 8,
 'Entity-Origin(e1,e2)': 9,
 'Entity-Origin(e2,e1)': 10,
 'Instrument-Agency(e1,e2)': 11,
 'Instrument-Agency(e2,e1)': 12,
 'Member-Collection(e1,e2)': 13,
 'Member-Collection(e2,e1)': 14,
 'Message-Topic(e1,e2)': 15,
 'Message-Topic(e2,e1)': 16,
 'Product-Producer(e1,e2)': 17,
 'Product-Producer(e2,e1)': 18
}


@Metric.register("semeval2010_task8_metric")
class SemEval2010Task8Metric(Metric):
    def __init__(self):
        self.pred = []
        self.gold = []
        self._ids_to_labels = {v: k for k, v in LABEL_MAP.items()}
        self._gold_regex = re.compile("official score is.+ F1 = ([0-9\.]+)% >>>")

    def __call__(self, predicted_label_ids, gold_label_ids):
        # predicted_label_ids = (batch_size, ) tensor with predicted ids
        # gold_label_ids = (batch_size, ) tensor with gold label ids

        for store, ids in [[self.pred, predicted_label_ids],
                           [self.gold, gold_label_ids]]:
            for i in ids.detach().cpu().numpy().tolist():
                store.append(self._ids_to_labels[i])

    def get_metric(self, reset: bool = False):
        # write out temp files to eval
        with tempfile.NamedTemporaryFile('w') as fgold, tempfile.NamedTemporaryFile('w') as fpred:
            fgold.write(''.join(["{}\t{}\n".format(i, e) for i, e in enumerate(self.gold)]))
            fgold.flush()
            fpred.write(''.join(["{}\t{}\n".format(i, e) for i, e in enumerate(self.pred)]))
            fpred.flush()

            # run perl script
            cmd = ['perl', 'bin/semeval2010_task8_scorer-v1.2.pl',
                   fpred.name, fgold.name]
            try:
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                output_str = output.decode("utf-8")
                f1_str = self._gold_regex.search(output_str).groups()[0]
                f1 = float(f1_str)
            except subprocess.CalledProcessError:
                f1 = -1.0

            # files are deleted when context manager closes

        if reset:
            self.pred = []
            self.gold = []

        return f1


@DatasetReader.register("semeval2010_task8")
class SemEval2010Task8Reader(DatasetReader):
    def __init__(self,
                 tokenizer_and_candidate_generator: TokenizerAndCandidateGenerator,
                 entity_masking: str = "segment",
                 lazy: bool = False):

        super().__init__(lazy=lazy)
        self.tokenizer = tokenizer_and_candidate_generator
        self.tokenizer.whitespace_tokenize = True
        self.entity_masking = entity_masking
        assert entity_masking in ("segment", "entity_markers")

    def text_to_instance(self, *inputs) -> Instance:
        raise NotImplementedError

    def _read(self, file_path: str) -> Iterable[Instance]:
        with JsonFile(cached_path(file_path), 'r') as fin:
            for example in fin:
                sentence = example['sentence']

                raw_tokens = self.tokenizer.bert_word_tokenizer.tokenize(sentence)
                tokens = []
                k = 0
                start_e1 = end_e1 = start_e2 = end_e2 = None
                while k < len(raw_tokens):
                    if raw_tokens[k:(k+3)] == ['<', 'e1', '>']:
                        start_e1 = len(tokens)
                        if self.entity_masking == 'entity_markers':
                            tokens.append(('[e1start]'))
                        k += 3
                    elif raw_tokens[k:(k+3)] == ['<', 'e2', '>']:
                        start_e2 = len(tokens)
                        if self.entity_masking == 'entity_markers':
                            tokens.append(('[e2start]'))
                        k += 3
                    elif raw_tokens[k:(k+4)] == ['<', '/', 'e1', '>']:
                        if self.entity_masking == 'entity_markers':
                            tokens.append(('[e1end]'))
                        end_e1 = len(tokens)
                        k += 4
                    elif raw_tokens[k:(k+4)] == ['<', '/', 'e2', '>']:
                        if self.entity_masking == 'entity_markers':
                            tokens.append(('[e2end]'))
                        end_e2 = len(tokens)
                        k += 4
                    else:
                        tokens.append(raw_tokens[k])
                        k += 1

                assert start_e1 is not None and end_e1 is not None and start_e2 is not None and end_e2 is not None

                tokens_and_candidates = self.tokenizer.tokenize_and_generate_candidates(' '.join(tokens))

                # set the segment ids
                # offsets is the beginning offset for each original token
                if self.entity_masking == 'segment':
                    offsets = [1] + tokens_and_candidates['offsets_a'][:-1]
                    segment_ids = list(tokens_and_candidates['segment_ids'])
                    for s, e, ii in [[start_e1, end_e1, 1], [start_e2, end_e2, 2]]:
                        ll = offsets[e] - offsets[s]
                        segment_ids[offsets[s]:offsets[e]] = [ii] * ll
                    tokens_and_candidates['segment_ids'] = segment_ids

                fields = self.tokenizer.convert_tokens_candidates_to_fields(tokens_and_candidates)

                fields['sentence_id'] = MetadataField(str(example['sent_id']))

                fields['label_ids'] = LabelField(LABEL_MAP[example['label']], skip_indexing=True)

                # get the indices of the entity starts
                offsets = [1] + tokens_and_candidates['offsets_a'][:-1]
                idx1_offset = offsets[start_e1]
                idx2_offset = offsets[start_e2]

                fields['index_a'] = LabelField(idx1_offset, skip_indexing=True)
                fields['index_b'] = LabelField(idx2_offset, skip_indexing=True)

                yield Instance(fields)
