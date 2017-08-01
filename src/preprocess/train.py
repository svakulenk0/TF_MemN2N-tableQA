# -*- coding: utf-8 -*-
'''
author: svakulenko
18 Jul 2017
'''

import numpy as np
from operator import mul

from keras.models import Sequential, Model, load_model
from keras.layers.embeddings import Embedding
from keras.layers import Input, dot, LSTM, Dense, TimeDistributed, Activation, RepeatVector, Dropout
from keras.utils import plot_model, to_categorical

from question_generator import generate_questions
from load_csv_into_rows import SAMPLE_CSV_FILE, DATA_DIR
from load_data import get_data, encode_data


HIDDEN_SIZE = 128
EMBEDDINGS_SIZE = 64
MODEL_PATH = './models/model.h5'


def one_hot_encode_data(data, dic, maxlenr=242, maxlenq=82, maxlena=5):
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



    # row_maxlen = data[0][0].shape[1]
    # question_maxlen = questions_train.shape[1]
    # answer_maxlen = answers_train.shape[1]
    # print 'Max length of a row:', row_maxlen
    # print 'Max length of a question:', question_maxlen
    # print 'Max length of an answer:', answer_maxlen

    print('Vectorization...')

    rows = np.zeros((nsamples, maxlenr, len_dic), dtype='int32')
    questions = np.zeros((nsamples, maxlenq, len_dic), dtype='int32')
    answers = np.zeros((nsamples, maxlena, len_dic), dtype='int32')


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


def categorical_accuracy(y_true, y_pred, mask=True):
    '''
    categorical_accuracy adjusted for padding mask
    '''
    # if mask is not None:
    print y_true
    print y_pred
    eval_shape = (reduce(mul, y_true.shape[:-1]), y_true.shape[-1])
    print eval_shape
    y_true_ = np.reshape(y_true, eval_shape)
    y_pred_ = np.reshape(y_pred, eval_shape)
    flat_mask = np.flatten(mask)
    comped = np.equal(np.argmax(y_true_, axis=-1),
                      np.argmax(y_pred_, axis=-1))
    ## not sure how to do this in tensor flow
    good_entries = flat_mask.nonzero()[0]
    return np.mean(np.gather(comped, good_entries))

    # else:
    #     return K.mean(K.equal(K.argmax(y_true, axis=-1),
    #                           K.argmax(y_pred, axis=-1)))


def make_character_embedding_layer(word_index):
    embeddings = get_embeddings()
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    return embedding_layer


def train_model(training_data, dic):

    len_dic = len(dic) + 1
    print 'Vocabulary size:', len_dic

    tables_train, questions_train, answers_train = training_data
    # tables_v, questions_v, answers_v = validation_data

    # show row sample
    # print tables_train[0]

    # show training data stats (in chars)
    row_maxlen = tables_train.shape[1]
    question_maxlen = questions_train.shape[1]
    answer_maxlen = len(answers_train[0])
    print 'Max length of a row:', row_maxlen
    print 'Max length of a question:', question_maxlen
    print 'Max length of an answer:', answer_maxlen

    # input placeholders
    table = Input((row_maxlen,), dtype='int32')
    question = Input((question_maxlen,), dtype='int32')

    # dictionary map embeddings matrix

    # embed_chars()

    # embedding_matrix = np.zeros((len_dic, len_dic))

    # for char, i in dic.items():
    #     embedding_matrix[i] = 1


    # network architecture

    # read table
    table_encoder = Sequential()
    table_encoder.add(Embedding(input_dim=len_dic,
                                output_dim=EMBEDDINGS_SIZE,
                                input_length=row_maxlen,
                                # weights = [embedding_matrix],
                                mask_zero=True,
                                trainable=False))
    # Encode the input character sequence using an rnn, producing an output of HIDDEN_SIZE
    # table_encoder.add(LSTM(EMBEDDINGS_SIZE))
    # table_encoder.add(Dropout(0.3))
    table_encoded = table_encoder(table)

    # read question
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=len_dic,
                                   output_dim=EMBEDDINGS_SIZE,
                                   input_length=question_maxlen,
                                   # weights = [embedding_matrix],
                                   mask_zero=True,
                                   trainable=False))
    # Encode the input character sequence using an rnn, producing an output of HIDDEN_SIZE
    # question_encoder.add(LSTM(EMBEDDINGS_SIZE, input_shape=(question_maxlen, len_dic)))
    # question_encoder.add(LSTM(EMBEDDINGS_SIZE))
    # question_encoder.add(Dropout(0.3))
    question_encoded = question_encoder(question)

    # match table and question
    match = dot([table_encoded, question_encoded], axes=(2, 2))
    match = Activation('softmax')(match)

    # generate answer
    # answer = Sequential()
    # print match.shape
    # answer = LSTM((HIDDEN_SIZE, answer_maxlen))(match)
    answer = LSTM(HIDDEN_SIZE)(match)
    # answer = Dropout(0.3)(answer)
    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    answer = RepeatVector(answer_maxlen)(answer)
    # answer = LSTM(HIDDEN_SIZE, return_sequences=True)(answer)
    answer = TimeDistributed(Dense(output_dim=len_dic))(answer)
    # answer = Dense(answer_maxlen)(answer)
    answer = Activation('softmax')(answer)
    # answer = LSTM(answer_maxlen)(answer)


    model = Model(inputs=[table, question], outputs=answer)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  sample_weight_mode='temporal',
                  # metrics=[categorical_accuracy])
                  # loss_weights = weights,
                  metrics=['categorical_accuracy'])

    # define the checkpoint
    # filepath="./models/tmp/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]
    
    model.summary()

    # one hot encode true answers
    # print answers_train
    true_answers = one_hot_encode(answers_train, len_dic)
    sample_weights = np.where(answers_train > 0, 1., 0.)
    # print true_answers
    # print sample_weights

    # print answers_train

    # train
    model.fit([tables_train, questions_train], true_answers,
              batch_size=32,
              epochs=2,
              shuffle=True,
              verbose=2,
              sample_weight=sample_weights,
              # callbacks=callbacks_list,
              validation_split=0.2)
              # validation_data=([tables_v, questions_v], answers_v))
    return model


def one_hot_encode(array, len_dic):
    return np.array([to_categorical(vector, num_classes=len_dic) for vector in array])


def test_one_hot_encode_data(file=SAMPLE_CSV_FILE):
    data, dic = get_data(file)
    print one_hot_encode_data(data, dic)


def test_train_model(file=SAMPLE_CSV_FILE):
    data, dic = get_data(file)
    rows, questions, answers = encode_data(data, dic)
    # rows, questions, answers = one_hot_encode_data(data, dic)
    print '#samples:', len(rows)
    # training_data = (rows[:split], questions[:split], answers[:split])
    # print '#samples for training:', len(training_data[0])
    # validation_data = (rows[split:], questions[split:], answers[split:])
    # print '#samples for validation:', len(validation_data[0])
    model = train_model((rows, questions, answers), dic)

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
    print prediction
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
    # test_one_hot_encode_data()
    test_train_model()
    check_model()
