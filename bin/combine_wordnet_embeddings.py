
import torch
import numpy as np
import h5py

from allennlp.models.archival import load_archive
from kb.common import JsonFile


# includes @@PADDING@@, @@UNKNOWN@@, @@MASK@@, @@NULL@@
NUM_EMBEDDINGS = 117663

def generate_wordnet_synset_vocab(entity_file, vocab_file):
    vocab = ['@@UNKNOWN@@']
    
    with JsonFile(entity_file, 'r') as fin:
        for node in fin:
            if node['type'] == 'synset':
                vocab.append(node['id'])

    vocab.append('@@MASK@@')
    vocab.append('@@NULL@@')

    with open(vocab_file, 'w') as fout:
        fout.write('\n'.join(vocab))


def extract_tucker_embeddings(tucker_archive, vocab_file, tucker_hdf5):
    archive = load_archive(tucker_archive)

    with open(vocab_file, 'r') as fin:
        vocab_list = fin.read().strip().split('\n')

    # get embeddings
    embed = archive.model.kg_tuple_predictor.entities.weight.detach().numpy()
    out_embeddings = np.zeros((NUM_EMBEDDINGS, embed.shape[1]))

    vocab = archive.model.vocab

    for k, entity in enumerate(vocab_list):
        embed_id = vocab.get_token_index(entity, 'entity')
        if entity in ('@@MASK@@', '@@NULL@@'):
            # these aren't in the tucker vocab -> random init
            out_embeddings[k + 1, :] = np.random.randn(1, embed.shape[1]) * 0.004
        elif entity != '@@UNKNOWN@@':
            assert embed_id != 1
            # k = 0 is @@UNKNOWN@@, and want it at index 1 in output
            out_embeddings[k + 1, :] = embed[embed_id, :]

    # write out to file
    with h5py.File(tucker_hdf5, 'w') as fout:
        ds = fout.create_dataset('tucker', data=out_embeddings)


def get_gensen_synset_definitions(entity_file, vocab_file, gensen_file):
    from gensen import GenSen, GenSenSingle

    gensen_1 = GenSenSingle(
        model_folder='./data/models',
        filename_prefix='nli_large_bothskip',
        pretrained_emb='./data/embedding/glove.840B.300d.h5'
    )
    gensen_1.eval()

    definitions = {}
    with open(entity_file, 'r') as fin:
        for line in fin:
            node = json.loads(line)
            if node['type'] == 'synset':
                definitions[node['id']] = node['definition']

    with open(vocab_file, 'r') as fin:
        vocab_list = fin.read().strip().split('\n')

    # get the descriptions
    sentences = [''] * NUM_EMBEDDINGS
    for k, entity in enumerate(vocab_list):
        definition = definitions.get(entity)
        if definition is None:
            assert entity in ('@@UNKNOWN@@', '@@MASK@@', '@@NULL@@')
        else:
            sentences[k + 1] = definition

    embeddings = np.zeros((NUM_EMBEDDINGS, 2048), dtype=np.float32)
    for k in range(0, NUM_EMBEDDINGS, 32):
        sents = sentences[k:(k+32)]
        reps_h, reps_h_t = gensen_1.get_representation(
            sents, pool='last', return_numpy=True, tokenize=True
        )
        embeddings[k:(k+32), :] = reps_h_t
        print(k)

    with h5py.File(gensen_file, 'w') as fout:
        ds = fout.create_dataset('gensen', data=embeddings)


def combine_tucker_gensen(tucker_hdf5, gensen_hdf5, all_file):
    with h5py.File(tucker_hdf5, 'r') as fin:
        tucker = fin['tucker'][...]

    with h5py.File(gensen_hdf5, 'r') as fin:
        gensen = fin['gensen'][...]

    all_embeds = np.concatenate([tucker, gensen], axis=1)
    all_e = all_embeds.astype(np.float32)

    with h5py.File(all_file, 'w') as fout:
        ds = fout.create_dataset('tucker_gensen', data=all_e)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_wordnet_synset_vocab', default=False, action="store_true")
    parser.add_argument('--entity_file', type=str)
    parser.add_argument('--vocab_file', type=str)

    parser.add_argument('--generate_gensen_embeddings', default=False, action="store_true")
    parser.add_argument('--gensen_file', type=str)

    parser.add_argument('--extract_tucker', default=False, action="store_true")
    parser.add_argument('--tucker_archive_file', type=str)
    parser.add_argument('--tucker_hdf5_file', type=str)

    parser.add_argument('--combine_tucker_gensen', default=False, action="store_true")
    parser.add_argument('--all_embeddings_file', type=str)

    args = parser.parse_args()


    if args.generate_wordnet_synset_vocab:
        generate_wordnet_synset_vocab(args.entity_file, args.vocab_file)
    elif args.generate_gensen_embeddings:
        get_gensen_synset_definitions(args.entity_file, args.vocab_file, args.gensen_file)
    elif args.extract_tucker:
        extract_tucker_embeddings(args.tucker_archive_file, args.vocab_file, args.tucker_hdf5_file)
    elif args.combine_tucker_gensen:
        combine_tucker_gensen(args.tucker_hdf5_file, args.gensen_file, args.all_embeddings_file)
    else:
        raise ValueError

