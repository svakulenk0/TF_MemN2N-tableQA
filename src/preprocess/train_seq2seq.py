# -*- coding: utf-8 -*-
'''
author: svakulenko
1 Aug 2017
'''

import seq2seq
from seq2seq.models import SimpleSeq2Seq

from load_data import get_data, encode_data, one_hot_encode_data
from load_csv_into_rows import SAMPLE_CSV_FILE
from models import Seq2SeqtableQA

# Model properties
BATCH_SIZE = 32
HIDDEN_SIZE = 128


def train(file=SAMPLE_CSV_FILE):

    # load data
    data, dic = get_data(file)
    # tables_train, questions_train, answers_train = encode_data(data, dic)
    tables_train, questions_train, answers_train = one_hot_encode_data(data, dic)

    # compute data stats
    print '#samples:', len(tables_train)
    len_dic = len(dic) + 1
    print 'Vocabulary size:', len_dic
    row_maxlen = tables_train.shape[1]
    question_maxlen = questions_train.shape[1]
    answer_maxlen = len(answers_train[0])
    print 'Max length of a row:', row_maxlen
    print 'Max length of a question:', question_maxlen
    print 'Max length of an answer:', answer_maxlen

    # compile model
    model = Seq2SeqtableQA(row_maxlen, question_maxlen, answer_maxlen, len_dic, HIDDEN_SIZE, BATCH_SIZE)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.summary()
    
    # train
    # for tables_batch, questions_batch, answers_batch in batch_data(data, dic, BATCH_SIZE):
    #     nn_model.fit([tables_batch, questions_batch], answers_batch, batch_size=BATCH_SIZE, nb_epoch=1, show_accuracy=True, verbose=1)
    # print tables_train
    model.fit([tables_train, questions_train], answers_train,
              batch_size=BATCH_SIZE,
              epochs=2,
              shuffle=True,
              verbose=2,
              validation_split=0.2)


if __name__ == '__main__':
    train()
