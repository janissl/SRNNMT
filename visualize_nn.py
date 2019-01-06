#!/usr/bin/env python3

import sys
from keras.models import model_from_json
from keras.utils import plot_model

from cfg import load


def main(io_argument_path='io_args.yml'):
    args = load(io_argument_path)

    with open(args['model_name'] + '.json', encoding='utf-8') as json:
        model_json = json.read()

    model = model_from_json(model_json)
    plot_model(model, to_file='{}.png'.format(args['model_name']), show_shapes=True, rankdir='BT')


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
