

from allennlp.models.archival import load_archive
from allennlp.data import DatasetReader, Vocabulary, DataIterator
from allennlp.nn.util import move_to_device
from allennlp.common import Params

import numpy as np

from kb.include_all import *

from kb.evaluation.semeval2010_task8 import LABEL_MAP


def write_for_official_eval(model_archive_file, test_file, output_file,
                            label_ids_to_label):
    archive = load_archive(model_archive_file)
    model = archive.model

    reader = DatasetReader.from_params(archive.config['dataset_reader'])

    iterator = DataIterator.from_params(Params({"type": "basic", "batch_size": 4}))
    vocab = Vocabulary.from_params(archive.config['vocabulary'])
    iterator.index_with(vocab)

    model.cuda()
    model.eval()

    instances = reader.read(test_file)
    predictions = []
    for batch in iterator(instances, num_epochs=1, shuffle=False):
        batch = move_to_device(batch, cuda_device=0)
        output = model(**batch)

        batch_labels = [
            label_ids_to_label[i]
            for i in output['predictions'].cpu().numpy().tolist()
        ]

        predictions.extend(batch_labels)

    to_write = ''.join(["{}\t{}\n".format(i + 8001, e) for i, e in enumerate(model.metrics[0].pred)])
    with open(output_file, 'w') as fout:
        fout.write(to_write)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_archive', type=str)
    parser.add_argument('--evaluation_file', type=str)
    parser.add_argument('--output_file', type=str)

    args = parser.parse_args()

    # int -> str
    label_ids_to_label = {v:k for k, v in LABEL_MAP.items()}

    write_for_official_eval(args.model_archive,
                            args.evaluation_file,
                            args.output_file,
                            label_ids_to_label)
