
# compute the heldout perplexity, next sentence prediction accuracy and loss

import tqdm

from allennlp.models.archival import load_archive
from allennlp.data import DatasetReader, DataIterator
from allennlp.common import Params
from allennlp.nn.util import move_to_device

from kb.include_all import BertPretrainedMaskedLM, KnowBert


def run_evaluation(evaluation_file, model_archive,
                   random_candidates=False):

    archive = load_archive(model_archive)
    model = archive.model
    vocab = model.vocab
    params = archive.config

    model.multitask = False
    model.multitask_kg = False
    model.cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    reader_params = params.pop('dataset_reader')
    if reader_params['type'] == 'multitask_reader':
        reader_params = reader_params['dataset_readers']['language_modeling']

    if random_candidates:
            for k, v in reader_params['base_reader']['tokenizer_and_candidate_generator']['entity_candidate_generators'].items():
                v['random_candidates'] = True

    reader = DatasetReader.from_params(Params(reader_params))

    iterator = DataIterator.from_params(Params({
            "type": "self_attn_bucket",
            "batch_size_schedule": "base-11gb-fp32",
            "iterator":{
                  "type": "bucket",
                  "batch_size": 32,
                  "sorting_keys": [["tokens", "num_tokens"]],
                  "max_instances_in_memory": 2500,
              }
    }))
    iterator.index_with(vocab)
    instances = reader.read(evaluation_file)

    for batch_no, batch in enumerate(tqdm.tqdm(iterator(instances, num_epochs=1))):
        b = move_to_device(batch, 0)
        loss = model(**b)
        if batch_no % 100 == 0:
            print(model.get_metrics())

    print(model.get_metrics())


if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluation_file', type=str)
    parser.add_argument('-m', '--model_archive', type=str)

    args = parser.parse_args()

    run_evaluation(args.evaluation_file,
                   model_archive=args.model_archive,
                   random_candidates=False)

