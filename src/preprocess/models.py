# -*- coding: utf-8 -*-
'''
author: svakulenko
1 Aug 2017
'''
from __future__ import absolute_import
from recurrentshop import LSTMCell, RecurrentSequential
from seq2seq.cells import LSTMDecoderCell, AttentionDecoderCell
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input, dot


def Seq2SeqtableQA(row_maxlen, question_maxlen, answer_maxlen, len_dic, hidden_dim, batch_size,
                   depth=(1,1), dropout=0.0, unroll=False, stateful=False):
# def Seq2SeqtableQA(output_dim, output_length, hidden_dim=None, input_shape=None,
               # batch_size=None, batch_input_shape=None, input_dim=None,
               # input_length=None, depth=1, dropout=0.0, unroll=False,
               # stateful=False):

    '''
    Based on SimpleSeq2Seq
    from https://github.com/farizrahman4u/seq2seq/blob/master/seq2seq/models.py
    '''

    # input placeholders
    table = Input((batch_size, row_maxlen))
    question = Input((batch_size, question_maxlen))

    # table encoder
    table_encoder = RecurrentSequential(unroll=unroll, stateful=stateful)
    # table_encoder.add(Embedding(input_dim=len_dic,
    #                             output_dim=EMBEDDINGS_SIZE,
    #                             input_length=row_maxlen,
    #                             # weights = [embedding_matrix],
    #                             mask_zero=True,
    #                             trainable=False))
    table_encoder.add(LSTMCell(hidden_dim, batch_input_shape=(batch_size, row_maxlen)))

    for _ in range(1, depth[0]):
        table_encoder.add(Dropout(dropout))
        table_encoder.add(LSTMCell(hidden_dim))

    table_encoded = table_encoder(table)


    # question encoder
    question_encoder = RecurrentSequential(unroll=unroll, stateful=stateful)
    question_encoder.add(LSTMCell(hidden_dim, batch_input_shape=(batch_size, question_maxlen)))

    for _ in range(1, depth[0]):
        question_encoder.add(Dropout(dropout))
        question_encoder.add(LSTMCell(hidden_dim))

    question_encoded = question_encoder(question)


    # match table and question
    match = dot([table_encoded, question_encoded], axes=(1, 1))
    # match = Activation('softmax')(match)

    # answer decoder
    answer_decoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                  decode=True, output_length=answer_maxlen)
    answer_decoder.add(Dropout(dropout, batch_input_shape=(batch_size, hidden_dim)))

    if depth[1] == 1:
        answer_decoder.add(LSTMCell(len_dic))
    else:
        answer_decoder.add(LSTMCell(hidden_dim))
        for _ in range(depth[1] - 2):
            answer_decoder.add(Dropout(dropout))
            answer_decoder.add(LSTMCell(hidden_dim))
    answer_decoder.add(Dropout(dropout))
    answer_decoder.add(LSTMCell(len_dic))

    answer_decoded = answer_decoder(match)


    return Model(inputs=[table, question], outputs=answer_decoded)
