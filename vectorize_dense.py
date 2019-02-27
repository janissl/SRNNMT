#!/usr/bin/env python3
"""Read source language and target language text file and turn these into dense vectors"""

import os
import sys
import itertools
from keras.models import model_from_json
import numpy as np

import data_dense
from cfg import load


# load model
def load_model(mname):
    with open(os.path.splitext(mname)[0] + ".json") as f:
        trained_model = model_from_json(f.read())
        trained_model.load_weights(mname + ".h5")
        trained_model.layers.pop()  # remove cosine and flatten layers
        trained_model.layers.pop()
        trained_model.outputs = [trained_model.get_layer('source_dense').output,
                                 trained_model.get_layer('target_dense').output]  # define new outputs
        trained_model.layers[-1].outbound_nodes = []  # not sure if we need these...
        trained_model.layers[-2].outbound_nodes = []
        print(trained_model.summary())
        print(trained_model.outputs)

    return trained_model


def fill_batch(data_size, max_sent_len, vs, data_iterator, ngrams):
    """ Iterate over the data_iterator and fill the index matrices with fresh data
        ms = matrices, vs = vocabularies
    """
    # custom fill_batch to return also sentences...
    ms = data_dense.Matrices(data_size, max_sent_len, ngrams)
    batchsize, max_sentence_len = ms.source_ngrams[ngrams[0]].shape  # just pick any one of these really
    row = 0
    src_sents = list()
    trg_sents = list()

    for (sent_src, sent_target), target in data_iterator:
        src_sents.append(sent_src)
        trg_sents.append(sent_target)

        for N in ngrams:
            for j, ngram in enumerate(data_dense.ngram_iterator(sent_src, N, max_sent_len)):
                ms.source_ngrams[N][row, j] = vs.get_id(ngram, vs.source_ngrams[N])
            for j, ngram in enumerate(data_dense.ngram_iterator(sent_target, N, max_sent_len)):
                ms.target_ngrams[N][row, j] = vs.get_id(ngram, vs.target_ngrams[N])

        ms.src_len[row] = len(sent_src.strip().split())
        ms.trg_len[row] = len(sent_target.strip().split())
        ms.targets[row] = target
        row += 1

        if row == batchsize:
            # print(ms.matrix_dict, ms.targets)
            yield ms.matrix_dict, ms.targets, src_sents, trg_sents
            src_sents = list()
            trg_sents = list()
            row = 0
            ms = data_dense.Matrices(data_size, max_sent_len, ngrams)
        else:
            if row > 0:
                yield ms.matrix_dict, ms.targets, src_sents, trg_sents


def iter_wrapper(src, trg, max_sent=0):
    counter = 0
    for src_sent, trg_sent in itertools.zip_longest(src, trg, fillvalue="#None#"):  # shorter padded with 'None'
        yield (src_sent, trg_sent), 1.0
        counter += 1
        if max_sent != 0 and counter == max_sent:
            break


def get_sentence_count(filepath):
    count = 0
    with open(filepath, 'rb') as file:
        while True:
            buffer = file.read(8192*1024)
            if not buffer:
                break
            count += buffer.count(b'\n')
    return count + 1


def vectorize(properties_path='props.yml', io_argument_path='io_args.yml'):
    # read and create files
    props = load(properties_path)
    args = load(io_argument_path)

    output_filename_extension = '.npy'

    print('Vectorizing all sentences', file=sys.stderr)

    # read vocabularies
    vs = data_dense.read_vocabularies('{}-vocab.pickle'.format(args['model_name']),
                                      "xxx",
                                      "xxx",
                                      False,
                                      props['ngram_length'])
    vs.trainable = False

    # load model
    trained_model = load_model('{}.{}'.format(args['model_name'], args['epoch_number']))
    output_size = trained_model.get_layer('source_dense').output_shape[1]
    max_sent_len = trained_model.get_layer('source_ngrams_{n}'.format(n=props['ngram_length'][0])).output_shape[1]
    print(output_size, max_sent_len, file=sys.stderr)

    # build matrices
    for entry in os.scandir(args['preprocessed_source_data_directory']):
        if not (entry.is_file() and entry.name.endswith('_{}.snt'.format(args['source_language']))):
            continue

        src_in_path = entry.path
        trg_in_path = entry.path.rsplit('_', 1)[0] + '_{}.snt'.format(args['target_language'])
        # trg_in_path = '{}.{}'.format(os.path.splitext(entry.path)[0], args['target_language'])

        if not os.path.exists(trg_in_path):
            continue

        src_out_path = entry.path + output_filename_extension
        trg_out_path = trg_in_path + output_filename_extension

        max_sent_count = max(get_sentence_count(src_in_path), get_sentence_count(trg_in_path))

        with open(src_in_path, encoding='utf-8') as src_inp, \
                open(trg_in_path, encoding='utf-8') as trg_inp, \
                open(src_out_path, 'wb') as src_outp, \
                open(trg_out_path, 'wb') as trg_outp:
            # get vectors
            counter = 0

            for i, (mx, targets, src_data, trg_data) in enumerate(fill_batch(max_sent_count,
                                                                             max_sent_len,
                                                                             vs,
                                                                             iter_wrapper(src_inp,
                                                                                          trg_inp),
                                                                             props['ngram_length'])):
                src, trg = trained_model.predict(mx)  # shape is (max_sent_count, props['gru_width'])
                # loop over items in batch
                for j, (src_v, trg_v) in enumerate(zip(src, trg)):
                    if j >= len(src_data):  # empty padding of the batch
                        break

                    write_vector_to_file(src_outp, normalize_v(src_v), src_data[j])
                    write_vector_to_file(trg_outp, normalize_v(trg_v), trg_data[j])

                    counter += 1

                    if counter > 0 and counter % 100 == 0:
                        print('{}: vectorized {} sentence pairs'.format(os.path.splitext(entry.name)[0], counter),
                              end='\r', file=sys.stderr, flush=True)

            print('{}: vectorized {} sentence pairs'.format(os.path.splitext(entry.name)[0], counter),
                  file=sys.stderr, flush=True)


def normalize_v(v):
    return v / np.linalg.norm(v)


def write_vector_to_file(file, vec, data):
    if data != '#None#':
        vec.astype(np.float32).tofile(file)


if __name__ == "__main__":
    sys.exit(vectorize(*sys.argv[1:]))
