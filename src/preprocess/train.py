# -*- coding: utf-8 -*-
'''
author: svakulenko
18 Jul 2017
'''

import cPickle as pkl
import numpy as np
from random import randrange

from keras.models import Sequential, Model, load_model
from keras.layers.embeddings import Embedding
from keras.layers import Input, dot, LSTM, Dense, TimeDistributed, Activation, RepeatVector, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model, to_categorical

from question_generator import generate_questions
from load_csv_into_rows import SAMPLE_CSV_FILE, DATA_DIR


HIDDEN_SIZE = 128
EMBEDDINGS_SIZE = 64
MODEL_PATH = './models/model.h5'


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
    encodes data as a sequence of characters using a dictionary
    for the input layer
    '''
    rows = []
    questions = []
    answers = []

    nqas = len(data[0][1])

    for row in data:
        # encode row
        rows.append([dic[c] for c in row[0]])

        # pick a sample qa 
        sample_qa = randrange(0, nqas)

        # encode sample qa
        question = row[1][sample_qa][0].strip('?')
        # print question
        questions.append([dic[c] for c in question])
        answers.append([dic[c] for c in row[1][sample_qa][1]])

    return (pad_sequences(rows, padding='post'),
            pad_sequences(questions, padding='post'),
            pad_sequences(answers, padding='post'))


def train_model(training_data, len_dic):
    tables_train, questions_train, answers_train = training_data
    # tables_v, questions_v, answers_v = validation_data

    # show row sample
    # print tables_train[0]

    # show training data stats (in chars)
    row_maxlen = tables_train.shape[1]
    question_maxlen = questions_train.shape[1]
    answer_maxlen = answers_train.shape[1]
    print 'Max length of a row:', row_maxlen
    print 'Max length of a question:', question_maxlen
    print 'Max length of an answer:', answer_maxlen

    # input placeholders
    table = Input((row_maxlen,), dtype='int32')
    question = Input((question_maxlen,), dtype='int32')


    # network architecture

    # read table
    table_encoder = Sequential()
    table_encoder.add(Embedding(input_dim=len_dic,
                                output_dim=EMBEDDINGS_SIZE,
                                input_length=row_maxlen))
    # Encode the input character sequence using an rnn, producing an output of HIDDEN_SIZE
    # table_encoder.add(LSTM(EMBEDDINGS_SIZE, input_shape=(row_maxlen, len_dic)))
    table_encoder.add(Dropout(0.3))
    table_encoded = table_encoder(table)

    # read question
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=len_dic,
                                   output_dim=EMBEDDINGS_SIZE,
                                   input_length=question_maxlen))
    # Encode the input character sequence using an rnn, producing an output of HIDDEN_SIZE
    # question_encoder.add(LSTM(EMBEDDINGS_SIZE, input_shape=(question_maxlen, len_dic)))
    question_encoder.add(Dropout(0.3))
    question_encoded = question_encoder(question)

    # match table and question
    match = dot([table_encoded, question_encoded], axes=(2, 2))
    match = Activation('softmax')(match)

    # generate answer
    # answer = Sequential()
    answer = LSTM(HIDDEN_SIZE)(match)
    answer = Dropout(0.3)(answer)
    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    answer = RepeatVector(answer_maxlen)(answer)
    # answer = LSTM(HIDDEN_SIZE, return_sequences=True)(answer)
    answer = TimeDistributed(Dense(len_dic))(answer)
    # answer = Dense(answer_maxlen)(answer)
    answer = Activation('softmax')(answer)
    # answer = LSTM(answer_maxlen)(answer)


    model = Model(inputs=[table, question], outputs=answer)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # define the checkpoint
    # filepath="./models/tmp/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]
    
    model.summary()

    # one hot encode true answers
    true_answers = np.array([to_categorical(answer, num_classes=len_dic) for answer in answers_train])

    # train
    model.fit([tables_train, questions_train], true_answers,
              batch_size=32,
              epochs=10,
              shuffle=True,
              # callbacks=callbacks_list,
              validation_split=0.2)
              # validation_data=([tables_v, questions_v], answers_v))

    return model


def test_encode_data(file=SAMPLE_CSV_FILE):
    data, dic = get_data(file)
    rows, questions, answers = encode_data(data[:2], dic)
    print rows


def test_train_model(file=SAMPLE_CSV_FILE, split=6000):
    data, dic = get_data(file)
    len_dic = len(dic) + 1
    print 'Vocabulary size:', len_dic
    rows, questions, answers = encode_data(data, dic)
    print '#samples:', len(rows)
    # training_data = (rows[:split], questions[:split], answers[:split])
    # print '#samples for training:', len(training_data[0])
    # validation_data = (rows[split:], questions[split:], answers[split:])
    # print '#samples for validation:', len(validation_data[0])
    model = train_model((rows, questions, answers), len_dic)

    model.save(MODEL_PATH)

    # evaluate the model
    # scores = model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    


def check_model(path=MODEL_PATH, file=SAMPLE_CSV_FILE, nsamples=2):
    '''
    see predictions generated for the training dataset
    '''

    # load model
    model = load_model(path)

    # load data
    data, dic = get_data(file)
    rows, questions, true_answers = encode_data(data, dic)

    # visualize model graph
    # plot_model(model, to_file='tableqa_model.png')

    # predict answers
    prediction = model.predict([rows[:nsamples], questions[:nsamples]])
    # print prediction
    predicted_answers = [[np.argmax(character) for character in sample] for sample in prediction]
    print predicted_answers
    print true_answers[:nsamples]

    # one hot encode answers
    # true_answers = [to_categorical(answer, num_classes=len(dic)) for answer in answers[:nsamples]]

    # decode chars from char ids int
    inv_dic = {v: k for k, v in dic.iteritems()}
    for i in xrange(nsamples):
        print '\n'
        # print 'Predicted answer: ' + ''.join([dic[char] for char in sample])
        print 'Table: ' + ''.join([inv_dic[char_id] for char_id in rows[i] if char_id != 0])
        print 'Question: ' + ''.join([inv_dic[char_id] for char_id in questions[i] if char_id != 0])
        print 'Answer(correct): ' + ''.join([inv_dic[char_id] for char_id in true_answers[i] if char_id != 0])
        print 'Answer(predicted): ' + ''.join([inv_dic[char_id] for char_id in predicted_answers[i] if char_id != 0])


if __name__ == '__main__':
    # test_train_model()
    check_model()
