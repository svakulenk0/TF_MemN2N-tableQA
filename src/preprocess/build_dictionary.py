# -*- coding: utf-8 -*-
'''
author: svakulenko
18 Jul 2017
'''

from collections import OrderedDict
import numpy as np
import cPickle as pkl

from load_csv_into_rows import *


def count_chars(file):
    """
    Build a character dictionary from a csv file
    """
    charcount = OrderedDict()

    with open(file, 'rb') as f:
        for cc in f.read().splitlines():
            chars = list(cc.lower())
            # print chars
            for c in chars:
                if c not in charcount:
                    charcount[c] = 0
                charcount[c] += 1

    chars = charcount.keys()
    freqs = charcount.values()
    sorted_idx = np.argsort(freqs)[::-1]

    chardict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        chardict[chars[sidx]] = idx + 1

    return chardict, charcount


def save_dictionary(mydict, count, loc):
    """
    Save a dictionary to the specified location
    """
    with open(loc, 'w') as f:
        pkl.dump(mydict, f)
        pkl.dump(count, f)


def build_dictionary(file_name=SAMPLE_CSV_FILE):
    """
    Build a character dictionary from a csv file
    """
    chardict, charcount = count_chars(DATA_DIR+file_name)
    print '#Chars=dictionary size:', len(chardict)
    print chardict

    loc = DATA_DIR + file_name[:-4] + '_dict.pkl'
    print 'Saving into:', loc
    save_dictionary(chardict, charcount, loc)


if __name__ == '__main__':
    build_dictionary()