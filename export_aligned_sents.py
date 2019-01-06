#!/usr/bin/env python3

import os
import sys
import csv

from cfg import load


def do_export(conf):
    lang_pair = '{}-{}'.format(conf['source_language'], conf['target_language'])
    aligned_filename_suffix = '.{}.aligned'.format(lang_pair)
    alignment_threshold = float(conf.get('alignment_threshold', 0))
    lang_pair_work_directory = os.path.join(conf['work_directory'], lang_pair)

    for entry in os.scandir(lang_pair_work_directory):
        if entry.is_file() and entry.name.endswith(aligned_filename_suffix):
            pair_title = entry.name[:-len(aligned_filename_suffix)]
            output_src_path = os.path.join(lang_pair_work_directory,
                                           '{}_{}.snt.aligned'.format(pair_title, conf['source_language']))
            output_trg_path = os.path.join(lang_pair_work_directory,
                                           '{}_{}.snt.aligned'.format(pair_title, conf['target_language']))

            with open(entry.path, encoding='utf-8') as aligned, \
                    open(output_src_path, 'w', encoding='utf-8', newline='\n') as src, \
                    open(output_trg_path, 'w', encoding='utf-8', newline='\n') as trg:
                reader = csv.reader(aligned, dialect=csv.excel_tab)

                for row in reader:
                    if len(row) == 3 and \
                            float(row[0]) >= alignment_threshold and \
                            row[1] and row[2]:
                        src.write('{}\n'.format(row[1]))
                        trg.write('{}\n'.format(row[2]))


def main(config_path='io_args.yml'):
    try:
        conf = load(config_path)
        do_export(conf)
    except Exception as ex:
        sys.stderr.write(repr(ex))
        return 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
