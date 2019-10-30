
import numpy as np
import codecs

import glob
import time

CACHE_SIZE = int(1e8)


def read_files(file_glob):
    # Our memory: read self.cache_size tokens, from which we will generate BERT instances.
    memory = [[]]
    n_tokens = 0

    t1 = time.time()
    all_file_names = glob.glob(file_glob)
    print("Found {} total files".format(len(all_file_names)))

    file_number = 0
    for fname in glob.glob(file_glob):
        file_number += 1
        print("reading file number {}, {}, total memory {}, total time {}".format(file_number, fname, n_tokens, time.time() - t1))
        with codecs.open(fname, 'r', encoding='utf8') as open_file:
            for sentence in open_file:
                words = sentence.strip().split()

                # Empty lines are used as document delimiters
                if len(words) == 0:
                    memory.append([])
                else:
                    memory[-1].append(words)
                    n_tokens += len(words)

                if n_tokens > CACHE_SIZE:
                    for document_index in range(len(memory)):
                        for instance in gen_bert_instances(memory, document_index):
                            yield instance

                    n_tokens = 0
                    memory = [[]]

    for document_index in range(len(memory)):
        for instance in gen_bert_instances(memory, document_index):
            yield instance


def gen_bert_instances(all_documents, document_index):
    """
    Create bert instances from a given document
    """
    word_to_wordpiece_ratio = 1.33
    document = all_documents[document_index]

    # First, defining target sequence length
    # [[0.9, 128], [0.05, 256], [0.04, 384], [0.01, 512]]
    randnum = np.random.random()
    if randnum < 0.9:
        if np.random.random() < 0.1:
            target_seq_length = np.random.randint(2, 128)
        else:
            target_seq_length = 128
    elif randnum < 0.95:
        target_seq_length = 256
    elif randnum < 0.99:
        target_seq_length = 384
    else:
        target_seq_length = 512

    word_target_seq_length = target_seq_length / word_to_wordpiece_ratio

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)

        current_length += len(segment)
        if i == len(document) - 1 or current_length >= word_target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = np.random.randint(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                next_sentence_label = 0
                if len(current_chunk) == 1 or np.random.random() < 0.5:
                    next_sentence_label = 1
                    target_b_length = word_target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = np.random.randint(0, len(all_documents))
                        if random_document_index != document_index and len(
                                        all_documents[random_document_index]
                        ) > 0:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = np.random.randint(0, len(random_document))
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                yield tokens_a, tokens_b, next_sentence_label

            current_chunk = []
            current_length = 0
        i += 1

def read_file_sample_nsp_write_shards(file_glob, output_prefix, num_output_files):
    output_files = []
    for k in range(num_output_files):
        fname = output_prefix + str(k) + '.txt'
        output_files.append(open(fname, 'w'))

    for tokens_a, tokens_b, label in read_files(file_glob):
        file_index = np.random.randint(0, num_output_files)
        line = "{}\t{}\t{}\n".format(label, ' '.join(tokens_a), ' '.join(tokens_b))
        output_files[file_index].write(line)

    for k in range(num_output_files):
        output_files[k].close()

