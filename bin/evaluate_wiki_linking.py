
# the wiki entity linking model

from kb.knowbert import BertPretrainedMaskedLM, KnowBert
from kb.bert_tokenizer_and_candidate_generator import BertTokenizerAndCandidateGenerator
from kb.bert_pretraining_reader import BertPreTrainingReader
from kb.include_all import ModelArchiveFromParams
from kb.include_all import TokenizerAndCandidateGenerator

from allennlp.data import DatasetReader, Vocabulary, DataIterator
from allennlp.models import Model
from allennlp.common import Params
import tqdm

from allennlp.nn.util import move_to_device

import torch
import copy

from allennlp.models.archival import load_archive

from allennlp.data import Instance
from allennlp.data.dataset import Batch



def run_evaluation(evaluation_file,
                   model_archive_file,
                   is_wordnet_and_wiki=False):
    archive = load_archive(model_archive_file)

    params = archive.config
    vocab = Vocabulary.from_params(params.pop('vocabulary'))

    model = archive.model
    model.cuda()
    model.eval()

    if is_wordnet_and_wiki:
        reader_params = Params({
                "type": "aida_wiki_linking",
                "entity_disambiguation_only": False,
                "entity_indexer": {
                    "type": "characters_tokenizer",
                    "namespace": "entity_wiki",
                    "tokenizer": {
                        "type": "word",
                        "word_splitter": {
                            "type": "just_spaces"
                        }
                    }
                },
                "extra_candidate_generators": {
                    "wordnet": {
                        "type": "wordnet_mention_generator",
                        "entity_file": "s3://allennlp/knowbert/wordnet/entities.jsonl"
                    }
                },
                "should_remap_span_indices": True,
                "token_indexers": {
                    "tokens": {
                        "type": "bert-pretrained",
                        "do_lowercase": True,
                        "max_pieces": 512,
                        "pretrained_model": "bert-base-uncased",
                        "use_starting_offsets": True,
                    }
                }
            })
    else:
        reader_params = Params({
            "type": "aida_wiki_linking",
            "entity_disambiguation_only": False,
            "token_indexers": {
                "tokens": {
                    "type": "bert-pretrained",
                    "pretrained_model": "bert-base-uncased",
                    "do_lowercase": True,
                    "use_starting_offsets": True,
                    "max_pieces": 512,
                },
            },
            "entity_indexer": {
               "type": "characters_tokenizer",
               "tokenizer": {
                   "type": "word",
                   "word_splitter": {"type": "just_spaces"},
               },
               "namespace": "entity",
            },
            "should_remap_span_indices": True,
        })

    if is_wordnet_and_wiki:
        cg_params = Params({
                        "type": "bert_tokenizer_and_candidate_generator",
                        "bert_model_type": "bert-base-uncased",
                        "do_lower_case": True,
                        "entity_candidate_generators": {
                            "wordnet": {
                                "type": "wordnet_mention_generator",
                                "entity_file": "s3://allennlp/knowbert/wordnet/entities.jsonl"
                            }
                        },
                        "entity_indexers": {
                            "wordnet": {
                                "type": "characters_tokenizer",
                                "namespace": "entity_wordnet",
                                "tokenizer": {
                                    "type": "word",
                                    "word_splitter": {
                                        "type": "just_spaces"
                                    }
                                }
                            }
                        }
                    })
        candidate_generator = TokenizerAndCandidateGenerator.from_params(cg_params)

    reader = DatasetReader.from_params(Params(reader_params))

    iterator = DataIterator.from_params(Params({"type": "basic", "batch_size": 16}))
    iterator.index_with(vocab)

    instances = reader.read(evaluation_file)

    for batch_no, batch in enumerate(iterator(instances, shuffle=False, num_epochs=1)):
        b = move_to_device(batch, 0)

        b['candidates'] = {'wiki': {
                'candidate_entities': b.pop('candidate_entities'),
                'candidate_entity_priors': b.pop('candidate_entity_prior'),
                'candidate_segment_ids': b.pop('candidate_segment_ids'),
                'candidate_spans': b.pop('candidate_spans')}}
        gold_entities = b.pop('gold_entities')
        b['gold_entities'] = {'wiki': gold_entities}

        if is_wordnet_and_wiki:
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
            bb = move_to_device(b, 0)
        else:
            bb = b

        loss = model(**bb)
        if batch_no % 100 == 0:
            print(model.get_metrics())

    print(model.get_metrics())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluation_file', type=str)
    parser.add_argument('-a', '--model_archive', type=str)
    parser.add_argument('--wiki_and_wordnet', action='store_true')

    args = parser.parse_args()

    run_evaluation(args.evaluation_file, args.model_archive, is_wordnet_and_wiki=args.wiki_and_wordnet)

