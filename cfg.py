#!/usr/bin/env python3

import yaml


def load(config_path):
    with open(config_path, encoding='utf-8') as cfg_file:
        cfg_str = cfg_file.read()
    cfg = yaml.load(cfg_str)
    return cfg
