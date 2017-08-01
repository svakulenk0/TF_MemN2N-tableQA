# -*- coding: utf-8 -*-
'''
author: svakulenko
1 Aug 2017
'''
import cPickle as pkl
from random import randrange

from keras.preprocessing.sequence import pad_sequences

from load_csv_into_rows import SAMPLE_CSV_FILE, DATA_DIR


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


if __name__ == '__main__':
    test_encode_data()
