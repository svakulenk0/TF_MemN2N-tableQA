# -*- coding: utf-8 -*-
'''
author: svakulenko
1 Aug 2017
'''
import cPickle as pkl
from random import randrange
import numpy as np

from keras.preprocessing.sequence import pad_sequences

from load_csv_into_rows import SAMPLE_CSV_FILE, DATA_DIR

# Dataset properties
NSAMPLES = 6188
N_CHARS = 46  # vocabulary size
MAX_ROW = 242
MAX_Q = 82
MAX_A = 5


def get_data(file_name):
    # load training data 
    loc = DATA_DIR + file_name[:-4] + '_data.pkl'
    with open(loc, 'r') as f:
        data = pkl.load(f)

    # load dictionary
    loc = DATA_DIR + file_name[:-4] + '_dict.pkl'
    with open(loc, 'r') as f:
        dic = pkl.load(f)
    return data, dic


def encode_data(data, dic):
    '''
    encodes data into a 2D array
    as a sequence of characters using a dictionary
    '''
    rows = []
    questions = []
    answers = []
    # weights = []

    nqas = len(data[0][1])
    len_dic = len(dic)

    for row in data:
        # encode row
        rows.append([dic[c] for c in row[0]])

        # pick a sample qa 
        sample_qa = randrange(0, nqas)

        # encode sample qa
        question = row[1][sample_qa][0].strip('?')
        # print question
        questions.append([dic[c] for c in question])
        # answers.append([dic[c] for c in row[1][sample_qa][1]])
        
        # one hot encode answers
        # for each char is answer
        # answer = []
        # for c in row[1][sample_qa][1]:
        #     # init with 0s
        #     char_vector = np.zeros((len_dic), dtype='int32')
        #     char_vector[dic[c]] = 1
        #     answer.append(char_vector)
        # answers.append(answer)
        answers.append([dic[c] for c in row[1][sample_qa][1]])
        # weights.append([1 for c in row[1][sample_qa][1]])

    return (pad_sequences(rows, padding='post'),
            pad_sequences(questions, padding='post'),
            pad_sequences(answers, padding='post'))


def test_encode_data(file=SAMPLE_CSV_FILE):
    data, dic = get_data(file)
    rows, questions, answers = encode_data(data[:2], dic)
    print rows


def one_hot_encode_data(data, dic, row_maxlen=MAX_ROW, question_maxlen=MAX_Q, answer_maxlen=MAX_A):
    '''
    encodes data into a 3D tensor
    each sample as a sequence of characters using a dictionary
    each charachter is one-hot-encoded according to its index in the dictionary
    
    data list of tuples (row, [(q, a)]) list of qa tuples per row 
    e.g. ('what was internal_mig_emigration in st. willibald in 2014?', '41')

    dic OrderedDict character-level e.g. [(';', 1), ('1', 2) ...]
    '''

    len_dic = len(dic) + 1
    print 'Vocabulary size:', len_dic

    nsamples = len(data)
    print "# Rows:", nsamples
    
    nqas = len(data[0][1])
    print "# QA tuples per row:", nqas

    print('Vectorization...')

    rows = np.zeros((nsamples, row_maxlen, len_dic), dtype='int32')
    questions = np.zeros((nsamples, question_maxlen, len_dic), dtype='int32')
    answers = np.zeros((nsamples, answer_maxlen, len_dic), dtype='int32')


    for i, row in enumerate(data):
        
        # encode row
        for t, char in enumerate(row[0]):
            rows[i, t, dic[char]] = 1

        # pick a sample qa 
        sample_qa = randrange(0, nqas)

        # encode sample qa
        question = row[1][sample_qa][0].strip('?')
        # print question
        for t, char in enumerate(question):
            questions[i, t, dic[char]] = 1
        
        answer = row[1][sample_qa][1]
        for t, char in enumerate(answer):
            answers[i, t, dic[char]] = 1

    return (rows, questions, answers)


def test_one_hot_encode_data(file=SAMPLE_CSV_FILE):
    data, dic = get_data(file)
    print one_hot_encode_data(data, dic)


if __name__ == '__main__':
    test_one_hot_encode_data()
