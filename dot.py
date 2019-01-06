#!/usr/bin/env python3


import os
import sys
import array
from csr_csc_dot import csr_csc_dot_f
import numpy as np
import re
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

from dictionary_baseline import build_dictionary
from cfg import load

# TODO: vector dimensionality is hard-coded to 150 (load_data),
#       and sentence lengths and src-trg length comparisons are hard-coded to 5-30 and 0-5 (align_slices)


def build_translation_matrix(translation_dictionary, word2idx_orig, word2idx_foreign):
    # return sparse matrix which translates orig words into foreign
    column_size = len(word2idx_foreign)  # foreign vocab size
    row = array.array('i')
    col = array.array('i')

    for word, idx in sorted(word2idx_orig.items(), key=lambda x: x[1]):
        word = word.lower()

        if word not in translation_dictionary:
            continue

        for translation in translation_dictionary[word]:
            translation = translation.lower()

            if translation not in word2idx_foreign:
                continue

            row.append(idx)
            col.append(word2idx_foreign[translation])

    sparse = coo_matrix(
        (np.ones(len(row), dtype=np.float32),
         (np.frombuffer(row, dtype=np.int32),
          np.frombuffer(col, dtype=np.int32))),
        shape=(len(word2idx_orig), column_size),
        dtype=np.float32)

    return sparse
    
    
def tokenize(text):
    words = [re.sub(r'^\W+|\W+$', '', word).lower() for word in text.split() if word.strip()]
    wc = len(words)

    for i in reversed(range(wc)):
        if not words[i]:
            words.remove(words[i])

    return words


def build_sparse_sklearn(data, word2idx):
    # sparse
    vectorizer = CountVectorizer(lowercase=True,
                                 binary=True,
                                 vocabulary=word2idx,
                                 analyzer="word",
                                 tokenizer=tokenize,
                                 dtype=np.float32)
    sparse = vectorizer.fit_transform(data)

    return sparse


def load_data(filepath, feature_count, gru_width):
    sentences = [sent.strip() for sent in open(filepath, encoding='utf-8')]
    vectors = np.fromfile(filepath + '.npy', np.float32)
    vectors = vectors.reshape(int(len(vectors) / feature_count), gru_width)
    return sentences, vectors


def main(properties_path='props.yml', io_argument_path='io_args.yml'):
    args = load(io_argument_path)
    props = load(properties_path)

    lang_pair = '{}-{}'.format(args['source_language'], args['target_language'])
    aligned_filename_suffix = '.{}.aligned'.format(lang_pair)
    lang_pair_work_directory = os.path.join(args['work_directory'], lang_pair)

    dictionary_path_base = os.path.join(args['dictionary_directory'], args['dictionary_name'])
    vocabulary_path_base = os.path.join(args['vocabulary_directory'], args['vocabulary_name'])

    src_lang_vocabulary_path = '{}.{}'.format(vocabulary_path_base, args['source_language'])
    trg_lang_vocabulary_path = '{}.{}'.format(vocabulary_path_base, args['target_language'])

    src2trg_dictionary = build_dictionary(dictionary_path_base + ".f2e", src_lang_vocabulary_path)
    trg2src_dictionary = build_dictionary(dictionary_path_base + ".e2f", trg_lang_vocabulary_path)

    word2idx_src = \
        {word.strip().lower(): i for i, word in enumerate(open(src_lang_vocabulary_path, encoding="utf-8"))}
    word2idx_trg = \
        {word.strip().lower(): i for i, word in enumerate(open(trg_lang_vocabulary_path, encoding="utf-8"))}

    src2trg_matrix = build_translation_matrix(src2trg_dictionary, word2idx_src, word2idx_trg).tocsr()
    trg2src_matrix = build_translation_matrix(trg2src_dictionary, word2idx_trg, word2idx_src).tocsr()

    if not os.path.exists(lang_pair_work_directory):
        os.makedirs(lang_pair_work_directory)

    for entry in os.scandir(os.path.join(args['source_data_directory'], 'snt')):
        if not (entry.is_file() and entry.name.endswith('_{}.snt'.format(args['source_language']))):
            continue

        pair_title = entry.name.rsplit('.', 1)[0].rsplit('_', 1)[0]
        trg_lang_filepath = os.path.join(os.path.dirname(entry.path),
                                         '{}_{}.snt'.format(pair_title, args['target_language']))

        if not os.path.exists(trg_lang_filepath):
            continue

        with open(os.path.join(lang_pair_work_directory, pair_title + aligned_filename_suffix),
                  'w', encoding='utf-8', newline='\n') as out_combined, \
                open(os.path.join(lang_pair_work_directory, '{}.{}.keras'.format(pair_title, lang_pair)),
                     'w', encoding='utf-8', newline='\n') as out_keras, \
                open(os.path.join(lang_pair_work_directory, '{}.{}.baseline'.format(pair_title, lang_pair)),
                     'w', encoding='utf-8', newline='\n') as out_baseline:
            src_sentences, src_vectors = load_data(entry.path, props['feature_count'], props['gru_width'])
            trg_sentences, trg_vectors = load_data(trg_lang_filepath, props['feature_count'], props['gru_width'])

            src_sparse = build_sparse_sklearn(src_sentences, word2idx_src)
            src_normalizer = np.array([len(set(s.split())) for s in src_sentences], dtype=np.float32)

            sparse_dot_out = np.zeros((len(src_sentences), len(trg_sentences)), dtype=np.float32)
            sparse_dot_out2 = np.zeros((len(src_sentences), len(trg_sentences)), dtype=np.float32)

            trg_sparse = build_sparse_sklearn(trg_sentences, word2idx_trg)
            trg_normalizer = np.array([len(set(s.split())) for s in trg_sentences], dtype=np.float32)
            trg_normalizer = trg_normalizer.reshape((1, len(trg_normalizer)))[:, len(trg_sentences) - 1]

            trg_translated_sparse = (trg_sparse * trg2src_matrix).tocsc()
            trg_translated_sparse.data = np.ones(len(trg_translated_sparse.data), dtype=np.float32)

            trg_sparse = trg_sparse.tocsc()

            sim_matrix = np.dot(src_vectors[:len(src_sentences), :], trg_vectors[:len(trg_sentences), :].T)

            csr_csc_dot_f(0,
                          len(src_sentences),
                          src_sparse,
                          trg_translated_sparse,
                          sparse_dot_out)

            np.divide(sparse_dot_out,
                      src_normalizer.reshape((len(src_normalizer), 1))[:len(src_sentences), :],
                      sparse_dot_out)  # normalize

            tmp = src_sparse[:len(src_sentences), :] * src2trg_matrix
            tmp.data = np.ones(len(tmp.data), dtype=np.float32)  # force to binary

            csr_csc_dot_f(0,
                          tmp.shape[0],
                          tmp,
                          trg_sparse,
                          sparse_dot_out2)

            np.divide(sparse_dot_out2,
                      trg_normalizer,
                      sparse_dot_out2)  # normalize

            # sum sparse_dot_out and sparse_dot_out2, write results to sparse_dot_out
            np.add(sparse_dot_out, sparse_dot_out2, sparse_dot_out)
            # sum all three, write results to sparse_dot_out2
            np.add(sim_matrix, sparse_dot_out, sparse_dot_out2)

            # now sim_matrix has dense similarities, sparse_dot_out has baseline similarities,
            #   and sparse_dot_out2 has combined similarities

            argmaxs_keras = np.argmax(sim_matrix, axis=1)
            argmaxs_baseline = np.argmax(sparse_dot_out, axis=1)
            argmaxs_combined = np.argmax(sparse_dot_out2, axis=1)

            # Print results
            for j in range(argmaxs_keras.shape[0]):  # all three should have the same shape
                # keras
                print(sim_matrix[j, argmaxs_keras[j]],
                      src_sentences[j],
                      trg_sentences[argmaxs_keras[j]],
                      sep="\t",
                      file=out_keras,
                      flush=True)
                # baseline
                print(sparse_dot_out[j, argmaxs_baseline[j]] / 2.0,
                      src_sentences[j],
                      trg_sentences[argmaxs_baseline[j]],
                      sep="\t",
                      file=out_baseline,
                      flush=True)
                # combined
                print(sparse_dot_out2[j, argmaxs_combined[j]] / 3.0,
                      src_sentences[j],
                      trg_sentences[argmaxs_combined[j]],
                      sep="\t",
                      file=out_combined,
                      flush=True)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
