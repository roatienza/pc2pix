'''Render point clouds from test dataset using pc2pix

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse

import sys
import json
from shapenet import get_split

import os
import datetime
from random import randint

def get_ply(split_file='data/splits.json'):
    js = get_split(split_file)
    return js

JSON_PATH = "data/all_exp.json"
TARGET_PATH = "data/all_norm.json"
# make pretty json by running:
# cat all_norm.json | python -m json.tool > all_exp_norm.json

if __name__ == '__main__':
    js = get_split(JSON_PATH)

    steps = 0
    datalen = { 'train': 0, 'test': 0}
    datasets = ('train', 'test')
    for dataset in datasets:
        for key in js.keys():
            # key eg 03001627
            data = js[key]
            tags = data[dataset]
            datalen[dataset] = max(datalen[dataset], len(tags))
    
    for dataset in datasets:
        print(dataset, datalen[dataset])

    start_time = datetime.datetime.now()
    target_json = {}
    for key in js.keys():
        # key eg 03001627
        data = js[key]
        fields = {}
        for dkey in data.keys():
            if dkey == 'name':
                fields[dkey] = data[dkey]
                continue
            tags = data[dkey]
            tags_len = len(tags)
            diff = datalen[dkey] - tags_len
            print(key, dkey, "diff: ", diff)
            for i in range(diff):
                rand_tag = tags[randint(0, tags_len)]
                tags.append(rand_tag)
            fields[dkey] = tags
            print(key, dkey, "newlen: ", len(tags))

        target_json[key] = fields

    with open(TARGET_PATH, 'w') as outfile:
        json.dump(target_json, outfile)

# tag eg fff29a99be0df71455a52e01ade8eb6a 

