#!/usr/bin/env python3

import os
import sys

from cfg import load
from dot import tokenize


def write_vocabulary(source_path, vocab_path_base):
    lang_ext = os.path.splitext(source_path)[1]
    vocab_location = os.path.dirname(vocab_path_base)

    if not os.path.exists(vocab_location):
        os.makedirs(vocab_location)

    with open(source_path, encoding='utf-8') as source, \
            open(vocab_path_base + lang_ext, 'w', encoding='utf-8', newline='\n') as vocab:
        seen = dict()

        for line in source:
            words = tokenize(line.strip())

            for word in words:
                try:
                    seen[word] += 1
                except KeyError:
                    vocab.write(word + '\n')
                    seen[word] = 1


def main(config_path='io_args.yml'):
    try:
        conf = load(config_path)
        vocab_path_base = os.path.join(conf['vocabulary_directory'], conf['vocabulary_name'])

        for path in (conf['src_train'], conf['trg_train']):
            write_vocabulary(path, vocab_path_base)
    except Exception as ex:
        sys.stderr.write(repr(ex))
        return 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
