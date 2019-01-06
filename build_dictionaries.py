#!/usr/bin/env python3

import os
import sys

from cfg import load


def write_dictionary():
    pass


def main(config_path='io_args.yml'):
    try:
        conf = load(config_path)
    except Exception as ex:
        sys.stderr.write(repr(ex))
        return 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
