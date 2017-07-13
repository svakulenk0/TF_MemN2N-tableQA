'''
author: svakulenko
13 Jul 2017
'''
import os
import numpy as np

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove/'
GLOVE_FILE = 'glove.twitter.27B.200d.txt'


def get_embeddings(path_embeddings):
    embeddings = {}
    with open(path_embeddings, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
    return embeddings


def test_load_glove():
    gloves = get_embeddings(os.path.join(GLOVE_DIR, GLOVE_FILE))
    print gloves['man']


if __name__ == "__main__":
    test_load_glove()