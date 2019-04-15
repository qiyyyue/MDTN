# -*- coding: utf-8 -*-
# /usr/bin/python3
from __future__ import print_function

from typing import List

from MultiChannelDTN.hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import os
import regex
from collections import Counter


def make_vocab(org_files: List[str], des_file: str):
    '''Constructs vocabulary.

    Args:
      org_files: A string List. Input file path.
      des_file: A string. Output file name.

    Writes vocabulary line by line to `des_file`
    '''
    words = []
    for file in org_files:
        rf = codecs.open(file, 'r', 'utf-8')
        for line in rf:
            words += line.strip().split('<#>')[1].strip().replace('\t', ' ').split()
    word2cnt = Counter(words)
    wf = codecs.open(des_file, 'w', 'utf-8')
    wf.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
    for word, cnt in word2cnt.most_common(len(word2cnt)):
        print(word, cnt)
        wf.write(u"{}\t{}\n".format(word, cnt))


if __name__ == '__main__':
    org_base_path = '../Data/train_news_data/2016_election'
    org_file_list = ['test.txt', 'train.txt', 'val.txt']
    for i in range(len(org_file_list)):
        org_file_list[i] = os.path.join(org_base_path, org_file_list[i])
    des_file = os.path.join(org_base_path, 'vocab_cnt.txt')
    make_vocab(org_file_list, des_file)