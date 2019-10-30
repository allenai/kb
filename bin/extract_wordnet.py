
SYNSET_RELATIONSHIP_TYPES = [
    'hypernyms', 'instance_hypernyms',
    # these are the inverse relationship to hypernyms
    #'hyponyms', 'instance_hyponyms',
    'member_holonyms', 'substance_holonyms', 'part_holonyms',
    # these are inverse of *holonyms
    #'member_meronyms', 'substance_meronyms', 'part_meronyms',
    'attributes',
    'entailments',
    'causes',
    'also_sees',
    'verb_groups',
    'similar_tos'
]

LEMMA_RELATIONSHIP_TYPES = [
    'antonyms',
    'hypernyms', 'instance_hypernyms',
    #hyponyms, instance_hyponyms
    'member_holonyms', 'substance_holonyms', 'part_holonyms',
    #member_meronyms, substance_meronyms, part_meronyms
    'topic_domains', 'region_domains', 'usage_domains',
    'attributes',
    'derivationally_related_forms',
    'entailments',
    'causes',
    'also_sees',
    'verb_groups',
    'similar_tos',
    'pertainyms'
]


def extract_wordnet_from_nltk(entity_output_file, relation_output_file):
    from nltk.corpus import wordnet as wn
    import json

    # each node is a synset or synset+lemma
    # synsets have POS
    # synsets have several lemmas associated with them
    #       each lemma is keyed by something like able%3:00:00::
    #       where string = lemma, first number is POS, then sense id
    #
    # in addition to the synset-synset and lemma-lemma relationships,
    # we will also add synset_lemma relationship for lemmas contained
    # in each synset
    with open(entity_output_file, 'w') as fent, \
         open(relation_output_file, 'w') as frel:

        for synset in wn.all_synsets():
            node = {
                'id': synset.name(),
                'pos': synset.pos(),
                'lemmas': [lem.key() for lem in synset.lemmas()],
                'examples': synset.examples(),
                'definition': synset.definition(),
                'type': 'synset',
            }
            fent.write(json.dumps(node) + '\n')
    
            # synset-synset relationships
            for relation in SYNSET_RELATIONSHIP_TYPES:
                entity2 = [rel_synset.name()
                           for rel_synset in getattr(synset, relation)()]
                for e2 in entity2:
                    frel.write('{}\t{}\t{}\n'.format(synset.name(), 'synset_' + relation, e2))

            # now get synset-lemma and lemma-lemma relationships
            for lemma in synset.lemmas():
                node = {
                    'id': lemma.key(),
                    'pos': synset.pos(),
                    'synset': synset.name(),
                    'type': 'lemma',
                    'count': lemma.count(),
                }
                fent.write(json.dumps(node) + '\n')

                frel.write('{}\t{}\t{}\n'.format(synset.name(), 'synset_lemma', lemma.key()))

                # lemma-lemma
                for relation in LEMMA_RELATIONSHIP_TYPES:
                    entity2 = [rel_lemma.key()
                           for rel_lemma in getattr(lemma, relation)()]
                    for e2 in entity2:
                        frel.write('{}\t{}\t{}\n'.format(synset.name(), 'lemma_' + relation, e2))


def split_wordnet_train_dev(relation_file, train_output_file, dev_output_file):
    import random
    train_pct = 0.99

    with open(relation_file, 'r') as fin:
        all_relations = fin.read().strip().split('\n')

    random.shuffle(all_relations)

    n_relations = len(all_relations)
    cutoff = int(train_pct * n_relations)

    with open(train_output_file, 'w') as fout:
        fout.write('\n'.join(all_relations[:cutoff]))
    with open(dev_output_file, 'w') as fout:
        fout.write('\n'.join(all_relations[cutoff:]))


def extract_gloss_examples_wordnet(entity_file, wic_root_dir, output_file, include_definitions=False):
    """
    WIC train: 4579 of 6330, 0.7233807266982623
    WIC dev: 931 of 1276, 0.7296238244514106
    WIC test: 2007 of 2800, 0.7167857142857142
    total examples, examples considered, examples lemma not found:  48247 45310 1024
    total definitions:  117659
    """
    import os
    from kb.common import JsonFile
    from kb.wordnet import WORDNET_TO_SEMCOR_POS_MAP
    import spacy
    from nltk.stem import PorterStemmer

    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    stemmer = PorterStemmer()

    # make set of unique sentences in WIC
    wic_unique_sentences = {}
    for split in ['train', 'dev', 'test']:
        wic_unique_sentences[split] = set()
        data_file_name = os.path.join(wic_root_dir, split, split + '.data.txt')
        with open(data_file_name, 'r') as fin:
            for instance in fin:
                sentence1, sentence2 = instance.strip().split('\t')[-2:]
                for sent in [sentence1, sentence2]:
                    # remove all whitespace and strip punctuation
                    s = ''.join(sent.strip().split()).lower().rstrip('.')
                    wic_unique_sentences[split].add(s)

    # read through the dump
    wic_keys = ['train', 'dev', 'test']
    wic_counts = {k: 0 for k in wic_keys}
    n_examples = 0
    n_definitions = 0
    n_examples_considered = 0
    n_examples_not_found = 0

    examples_to_write = []

    nn = 0
    with JsonFile(entity_file, 'r') as fin:
        for entity in fin:
            nn += 1
            if nn % 100 == 0:
                print(nn)

            if entity['type'] == 'lemma':
                continue

            pos = WORDNET_TO_SEMCOR_POS_MAP[entity['pos']]

            if len(entity['definition']) > 0:
                n_definitions += 1
                if include_definitions:
                    doc = [t for t in nlp(entity['definition']) if not t.is_space]
                    lemma_id = entity['lemmas'][0]
                    lemma = lemma_id.partition('%')[0].replace('_', ' ').replace('-', ' - ').split(' ')
                    ex = lemma + ['is', 'defined', 'as', ':'] + [t.text for t in doc]
                    examples_to_write.append([' '.join(ex), (0, len(lemma)), lemma_id, pos])

            for example in entity['examples']:
                s = ''.join(example.strip().split()).lower().rstrip('.')
                n_examples += 1
                skip = False
                for key in wic_keys:
                    if s in wic_unique_sentences[key]:
                        wic_counts[key] += 1
                        if key == 'test' or key == 'dev':
                            skip = True

                if not skip:
                    # get the location of the lemma in the example
                    doc = [t for t in nlp(example) if not t.is_space]
                    n_examples_considered += 1

                    # need to check all the lemmas
                    found = False
                    for lemma_id in entity['lemmas']:

                        lemma = lemma_id.partition('%')[0].replace('_', ' ').replace('-', ' - ').split(' ')
                        len_lemma = len(lemma)

                        # exact match to word, exact match to lemma, word prefix, word suffix
                        lemma_indices = [[], [], []]
                        for i, t in enumerate(doc):
                            span = [t.text.lower() for t in doc[i:(i+len_lemma)]]
                            if span == lemma:
                                lemma_indices[0].append(i)
                            span_lemma = [t.lemma_.lower() for t in doc[i:(i+len_lemma)]]
                            if span_lemma == lemma:
                                lemma_indices[1].append(i)
                            if [stemmer.stem(t) for t in span] == [stemmer.stem(t) for t in lemma]:
                                lemma_indices[2].append(i)
    
                        # get the index
                        index = None
                        for ii in range(3):
                            if len(lemma_indices[ii]) == 1:
                                index = lemma_indices[ii][0]
                                break
                            elif len(lemma_indices[ii]) > 1:
                                break

                        if index is not None:
                            found = True
                            break

                    if found:
                        examples_to_write.append([' '.join(t.text for t in doc), (index, index+len(lemma)), lemma_id, pos])
                    else:
                        n_examples_not_found += 1

    for key in wic_counts:
        print("WIC {}: {} of {}, {}".format(key, wic_counts[key], len(wic_unique_sentences[key]),
                                            wic_counts[key] / len(wic_unique_sentences[key])))
    print("total examples, examples considered, examples lemma not found: ", n_examples, n_examples_considered, n_examples_not_found)
    print("total definitions: ", n_definitions)

    # WRITE OUT TO FILE!!
    with JsonFile(output_file, 'w') as fout:
        for ei, e_write in enumerate(examples_to_write):
            tokens = e_write[0].split()
            start, end = e_write[1]
            out = []
            for i in range(0, start):
                t = {'token': tokens[i], 'pos': '', 'lemma': ''}
                out.append(t)
    
            t = {'token': ' '.join(tokens[start:end]),
                 'pos': e_write[3],
                 'lemma': e_write[2].partition('%')[0],
                 'senses': [e_write[2].partition('%')[2]],
                 'id': 'example_definition.{}'.format(ei)}
            out.append(t)
    
            for i in range(end, len(tokens)):
                t = {'token': tokens[i], 'pos': '', 'lemma': ''}
                out.append(t)
    
            fout.write(out)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract wordnet from nltk')

    parser.add_argument('--extract_graph', default=False, action="store_true")
    parser.add_argument('--split_wordnet', default=False, action="store_true")
    parser.add_argument('--extract_examples_wordnet', default=False, action="store_true")

    parser.add_argument('--entity_file', type=str)
    parser.add_argument('--relationship_file', type=str)
    parser.add_argument('--relationship_train_file', type=str)
    parser.add_argument('--relationship_dev_file', type=str)
    
    parser.add_argument('--wic_root_dir', type=str)
    parser.add_argument('--wordnet_example_file', type=str)

    args = parser.parse_args()

    if args.extract_graph:
        extract_wordnet_from_nltk(args.entity_file, args.relationship_file)
    elif args.split_wordnet:
        split_wordnet_train_dev(args.relationship_file, args.relationship_train_file, args.relationship_dev_file)
    elif args.extract_examples_wordnet:
        extract_gloss_examples_wordnet(args.entity_file, args.wic_root_dir, args.wordnet_example_file)
    else:
        raise ValueError

