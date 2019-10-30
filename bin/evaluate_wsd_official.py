
import json
import copy

from allennlp.models.archival import load_archive
from allennlp.data import Instance
from allennlp.data.dataset import Batch

from kb.include_all import WordNetFineGrainedSenseDisambiguationReader
from kb.include_all import TokenizerAndCandidateGenerator

from allennlp.data import DatasetReader, Vocabulary
from allennlp.data.iterators import BasicIterator

from allennlp.common import Params
from allennlp.nn.util import move_to_device

from kb.common import JsonFile
from kb.knowbert import KnowBert


def make_files_for_official_eval(model_archive_file, evaluation_files, output_file,
                                 cuda_device):

    archive = load_archive(model_archive_file)
    model = archive.model

    model.eval()
    if cuda_device != -1:
        model.cuda(cuda_device)

    def find_key(d, func):
        ret = None
        stack = [d]
        while len(stack) > 0 and ret is None:
            s = stack.pop()
            for k, v in s.items():
                if func(k, v):
                    ret = s
                    break
                elif isinstance(v, dict):
                    stack.append(v)
        return ret

    # load reader
    full_reader_params = copy.deepcopy(archive.config['dataset_reader'].as_dict())
    reader_params = find_key(full_reader_params,
                             lambda k, v: k == 'type' and v == 'wordnet_fine_grained')
    reader_params['is_training'] = False
    reader_params['should_remap_span_indices'] = True
    if 'extra_candidate_generators' in reader_params:
        candidate_generator_params = find_key(
                full_reader_params,
                lambda k, v: k == 'tokenizer_and_candidate_generator'
        )['tokenizer_and_candidate_generator']
        candidate_generator = TokenizerAndCandidateGenerator.from_params(
                Params(candidate_generator_params)
        )

    reader_params = Params(reader_params)

    print("====================")
    print(reader_params.as_dict())
    print("====================")

    reader = DatasetReader.from_params(reader_params)

    synset_to_lemmas = {}
    for lemma_id, synset_id in reader.mention_generator._lemma_to_synset.items():
        if synset_id not in synset_to_lemmas:
            synset_to_lemmas[synset_id] = []
        synset_to_lemmas[synset_id].append(lemma_id)

    vocab_params = archive.config['vocabulary']
    vocab = Vocabulary.from_params(vocab_params)

    iterator = BasicIterator(batch_size=24)
    iterator.index_with(vocab)

    fout = open(output_file, 'w')

    for ds_file in [evaluation_file]:
        instances = reader.read(ds_file)

        # get the metadata ids from the raw file
        raw_lines = []
        with JsonFile(ds_file, 'r') as fin:
            for sentence in fin:
                raw_ids = [[token['id'], token['lemma']] for token in sentence if 'senses' in token]
                if len(raw_ids) > 0:
                    raw_lines.append(raw_ids)

        raw_i = 0
        for batch in iterator(instances, num_epochs=1, shuffle=False):
            print(raw_i)

            if cuda_device > -1:
                b = move_to_device(batch, cuda_device)
            else:
                b = batch

            b['candidates'] = {'wordnet': {
                    'candidate_entities': b.pop('candidate_entities'),
                    'candidate_entity_priors': b.pop('candidate_entity_prior'),
                    'candidate_segment_ids': b.pop('candidate_segment_ids'),
                    'candidate_spans': b.pop('candidate_spans')}}
            gold_entities = b.pop('gold_entities')
            b['gold_entities'] = {'wordnet': gold_entities}

            if 'extra_candidates' in b:
                extra_candidates = b.pop('extra_candidates')
                seq_len = b['tokens']['tokens'].shape[1]
                bbb = []
                for e in extra_candidates:
                    for k in e.keys():
                        e[k]['candidate_segment_ids'] = [0] * len(e[k]['candidate_spans'])
                    ee = {'tokens': ['[CLS]'] * seq_len, 'segment_ids': [0] * seq_len,
                          'candidates': e}
                    ee_fields = candidate_generator.convert_tokens_candidates_to_fields(ee)
                    bbb.append(Instance(ee_fields))
                eb = Batch(bbb)
                eb.index_instances(vocab)
                padding_lengths = eb.get_padding_lengths()
                tensor_dict = eb.as_tensor_dict(padding_lengths)
                b['candidates'].update(tensor_dict['candidates'])

            if cuda_device > -1:
                b = move_to_device(b, cuda_device)

            output = model(**b)
    
            # predicted entities is list of (batch_index, (start, end), entity_id)
            predicted_entities = model.soldered_kgs['wordnet'].entity_linker._decode(
                          output['wordnet']['linking_scores'], b['candidates']['wordnet']['candidate_spans'], 
                          b['candidates']['wordnet']['candidate_entities']['ids']
            )

            # make output file
            predicted_entities_batch_indices = []
            batch_size = batch['tokens']['tokens'].shape[0]
            for k in range(batch_size):
                predicted_entities_batch_indices.append([])
            for b_index, start_end, eid in predicted_entities:
                try:
                    synset_id = vocab.get_token_from_index(eid, 'entity')
                except KeyError:
                    synset_id = vocab.get_token_from_index(eid, 'entity_wordnet')
                all_lemma_ids = synset_to_lemmas[synset_id]
                predicted_entities_batch_indices[b_index].append(all_lemma_ids)

            # output lines look like semeval2013.d000.s001.t003 reader%1:19:00::
            for k in range(batch_size):
                raw_ids = raw_lines[raw_i]
                predicted_lemmas = predicted_entities_batch_indices[k]
                assert len(predicted_lemmas) == len(raw_ids)
                for (ii, gold_lemma), pl in zip(raw_ids, predicted_lemmas):
                    # get the predicted lemma_id
                    predicted_lemma_id = None
                    for pp in pl:
                        if pp.partition('%')[0] == gold_lemma:
                            predicted_lemma_id = pp
                    assert predicted_lemma_id is not None
                    line = "{} {}\n".format(ii, predicted_lemma_id)
                    fout.write(line)
                raw_i += 1

    fout.close()


if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_archive', type=str)
    parser.add_argument('--evaluation_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--cuda_device', type=int, default=-1)

    args = parser.parse_args()

    evaluation_file = args.evaluation_file

    make_files_for_official_eval(args.model_archive, evaluation_file, args.output_file,
                                 args.cuda_device)

