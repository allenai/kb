
from typing import Union, List

from allennlp.common import Params
from allennlp.data import Instance, DataIterator, Vocabulary

from kb.include_all import TokenizerAndCandidateGenerator
from kb.bert_pretraining_reader import replace_candidates_with_mask_entity

import json


def _extract_config_from_archive(model_archive):
    import tarfile
    with tarfile.open(model_archive, 'r:gz') as archive:
        config = Params(json.load(archive.extractfile('config.json')))
    return config


def _find_key(d, key):
    val = None
    stack = [d.items()]
    while len(stack) > 0 and val is None:
        s = stack.pop()
        for k, v in s:
            if k == key:
                val = v
                break
            elif isinstance(v, dict):
                stack.append(v.items())
    return val


class KnowBertBatchifier:
    """
    Takes a list of sentence strings and returns a tensor dict usable with
    a KnowBert model
    """
    def __init__(self, model_archive, batch_size=32,
                       masking_strategy=None,
                       wordnet_entity_file=None, vocab_dir=None):

        # get bert_tokenizer_and_candidate_generator
        config = _extract_config_from_archive(model_archive)

        # look for the bert_tokenizers and candidate_generator
        candidate_generator_params = _find_key(
            config['dataset_reader'].as_dict(), 'tokenizer_and_candidate_generator'
        )

        if wordnet_entity_file is not None:
            candidate_generator_params['entity_candidate_generators']['wordnet']['entity_file'] = wordnet_entity_file

        self.tokenizer_and_candidate_generator = TokenizerAndCandidateGenerator.\
                from_params(Params(candidate_generator_params))
        self.tokenizer_and_candidate_generator.whitespace_tokenize = False

        assert masking_strategy is None or masking_strategy == 'full_mask'
        self.masking_strategy = masking_strategy

        # need bert_tokenizer_and_candidate_generator
        if vocab_dir is not None:
            vocab_params = Params({"directory_path": vocab_dir})
        else:
            vocab_params = config['vocabulary']
        self.vocab = Vocabulary.from_params(vocab_params)

        self.iterator = DataIterator.from_params(
            Params({"type": "basic", "batch_size": batch_size})
        )
        self.iterator.index_with(self.vocab)

    def _replace_mask(self, s):
        return s.replace('[MASK]', ' [MASK] ')

    def iter_batches(self, sentences_or_sentence_pairs: Union[List[str], List[List[str]]]):
        # create instances
        instances = []
        for sentence_or_sentence_pair in sentences_or_sentence_pairs:
            if isinstance(sentence_or_sentence_pair, list):
                assert len(sentence_or_sentence_pair) == 2
                tokens_candidates = self.tokenizer_and_candidate_generator.\
                        tokenize_and_generate_candidates(
                                self._replace_mask(sentence_or_sentence_pair[0]),
                                self._replace_mask(sentence_or_sentence_pair[1]))
            else:
                tokens_candidates = self.tokenizer_and_candidate_generator.\
                        tokenize_and_generate_candidates(self._replace_mask(sentence_or_sentence_pair))

            print(self._replace_mask(sentence_or_sentence_pair))
            print(tokens_candidates['tokens'])

            # now modify the masking if needed
            if self.masking_strategy == 'full_mask':
                # replace the mask span with a @@mask@@ span
                masked_indices = [index for index, token in enumerate(tokens_candidates['tokens'])
                      if token == '[MASK]']

                spans_to_mask = set([(i, i) for i in masked_indices])
                replace_candidates_with_mask_entity(
                        tokens_candidates['candidates'], spans_to_mask
                )

                # now make sure the spans are actually masked
                for key in tokens_candidates['candidates'].keys():
                    for span_to_mask in spans_to_mask:
                        found = False
                        for span in tokens_candidates['candidates'][key]['candidate_spans']:
                            if tuple(span) == tuple(span_to_mask):
                                found = True
                        if not found:
                            tokens_candidates['candidates'][key]['candidate_spans'].append(list(span_to_mask))
                            tokens_candidates['candidates'][key]['candidate_entities'].append(['@@MASK@@'])
                            tokens_candidates['candidates'][key]['candidate_entity_priors'].append([1.0])
                            tokens_candidates['candidates'][key]['candidate_segment_ids'].append(0)
                            # hack, assume only one sentence
                            assert not isinstance(sentence_or_sentence_pair, list)


            fields = self.tokenizer_and_candidate_generator.\
                convert_tokens_candidates_to_fields(tokens_candidates)

            instances.append(Instance(fields))


        for batch in self.iterator(instances, num_epochs=1, shuffle=False):
            yield batch

