# -*- coding: utf-8 -*-
'''
author: svakulenko
13 Jul 2017
'''
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from settings import *


DATA_DIR = './data/'
SAMPLE_CSV_FILE = 'OOE_Wanderungen_Zeitreihe.csv'


def load_data(file_name):
    vocabulary = {}

    with open(DATA_DIR+file_name, 'rb') as f:
        for line_idx, line in enumerate(f):
            line = line.rstrip().lower()
            words = line.split()
            # print words

            # Skip substory index
            for k in range(1, len(words)):
                w = words[k]
                # print w

                if w.endswith('.') or w.endswith('?'):
                    w = w[:-1]
                if w not in vocabulary:
                    # add word to the vocabulary
                    vocabulary[w] = len(vocabulary)
    # print vocabulary
    print len(vocabulary)
    return vocabulary


def get_training_and_validation_sets(file_name=SAMPLE_CSV_FILE):
    header, rows = load_csv(file_name)
    cells, stories_txt = csv_2_triples(header, rows)
    stories, questions, word_index = tokenize_data(stories_txt)
    # X_processed, Y_processed, word_index = load_data()
    x_train, x_val, y_train, y_val = split_the_data(X_processed, Y_processed)
    return word_index, x_train, x_val, y_train, y_val


def csv_2_triples(header, rows):
    '''
    Embed tables as triples, e.g. 
    [[   0    0 7028    9   10]
     [   0 7028    4    8  653]
     [   0 7028    4   12  458]
     ..., 
     [6328    5    1    3  124]
     [6328    7    1    3   25]
     [   0 6328    3    6  143]]
    '''
    # ncells = len(rows) * len(header)
    # print ncells
    # table = np.zeros((ncells, 3), np.int16)
    table = []
    # print table
    vocabulary = {"nil": 0}
    # add headers to vocabulary
    for column_name in header:
        vocabulary[column_name] = len(vocabulary)
    # cell counter
    # n = 0
    for i, row in enumerate(rows):
        # print row
        row_id = 'row' + str(i)
        vocabulary[row_id] = len(vocabulary)
        for j, cell_value in enumerate(row):
            # stories_txt.append(';'.join([row_id, header[j], cell_value]))
            if cell_value not in vocabulary:
                vocabulary[cell_value] = len(vocabulary)
            # embed row
            # print row_id, header[j], cell_value
            # print [vocabulary[row_id], vocabulary[header[j]], vocabulary[cell_value]]
            table.append([vocabulary[row_id], vocabulary[header[j]], vocabulary[cell_value]])
            # n += 1
    # print n
    # 61880
    # print table
    return vocabulary, np.array(table)


def tokenize_data(texts):
    '''
    texts - array of strings to tokenize
    '''
    # print texts
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, split=";")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    # print sequences
    word_index = tokenizer.word_index
    stories = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return stories, word_index


def test_tokenize_data():
    header, rows = load_csv(SAMPLE_CSV_FILE)
    cells, stories_txt = csv_2_triples(header, rows)
    stories, word_index = tokenize_data(stories_txt)
    print stories


def test_csv_2_triples():
    header, rows = load_csv(SAMPLE_CSV_FILE)
    vocabulary, table = csv_2_triples(header, rows)
    print table


def test_load_data():
    sample_data_file = 'table_data_train.txt'
    load_data(sample_data_file)


if __name__ == "__main__":
    test_load_csv()