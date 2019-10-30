'''
Original data from: http://lcl.uniroma1.it/wsdeval/

Data converted from XML to jsonl
'''

import os
import h5py
import json

import numpy as np

from kb.common import JsonFile

def read_gold_data(fname):
    gold_data = {}
    with open(fname, 'r') as fin:
        for line in fin:
            ls = line.strip().split()
            lid = ls[0]
            if lid not in gold_data:
                gold_data[lid] = set()
            for sense in ls[1:]:
                gold_data[lid].add(sense)
    return gold_data


def read_wsd_data(fname, fname_gold):
    from lxml import etree

    gold_data = read_gold_data(fname_gold)

    with open(fname, 'r') as fin:
        data = fin.read()

    corpus = etree.fromstring(data.encode('utf-8'))

    sentences = []
    n_sentences = 0
    for node in corpus.iterdescendants():
        if node.tag == 'sentence':
            sentence = []
            for token_node in node.iterdescendants():
                token = {
                    'token': token_node.text,
                    'lemma': token_node.attrib['lemma'],
                    'pos': token_node.attrib['pos']
                }
                if token_node.tag == 'instance':
                    token['id'] = token_node.attrib['id']
                    token['senses'] = []
                    for sense in gold_data[token['id']]:
                        lemma, _, ss = sense.partition('%')
                        assert lemma == token['lemma']
                        token['senses'].append(ss)

                sentence.append(token)

            sentences.append(sentence)

    return sentences


def get_dataset_metadata(wsd_framework_root):
    return [
            [
                'semcor',
                os.path.join(
                    wsd_framework_root, 'Training_Corpora', 'SemCor', 'semcor'
                )
            ], [
                'senseval2',
                os.path.join(
                    wsd_framework_root, 'Evaluation_Datasets', 'senseval2',
                    'senseval2'
                )
            ], [
                'senseval3',
                os.path.join(
                    wsd_framework_root, 'Evaluation_Datasets', 'senseval3',
                    'senseval3'
                )
            ], [
                'semeval2015',
                os.path.join(
                    wsd_framework_root, 'Evaluation_Datasets', 'semeval2015',
                    'semeval2015'
                )
            ], [
                'semeval2013',
                os.path.join(
                    wsd_framework_root, 'Evaluation_Datasets', 'semeval2013',
                    'semeval2013'
                )
            ], [
                'semeval2007',
                os.path.join(
                    wsd_framework_root, 'Evaluation_Datasets', 'semeval2007',
                    'semeval2007'
                )
            ]
        ]


def convert_all_wsd_datasets(outdir, wsd_framework_root):
    datasets = get_dataset_metadata(wsd_framework_root)

    for ds in datasets:
        ds_name, ds_root = ds
        data = read_wsd_data(ds_root + '.data.xml', ds_root + '.gold.key.txt')
        with JsonFile(os.path.join(outdir, ds_name + '.json'), 'w') as fout:
            for line in data:
                fout.write(line)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--wsd_framework_root', type=str)
    parser.add_argument('--outdir', type=str)

    args = parser.parse_args()

    convert_all_wsd_datasets(args.outdir, args.wsd_framework_root)

