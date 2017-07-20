# -*- coding: utf-8 -*-
'''
author: svakulenko
18 Jul 2017
'''
from random import randrange
import cPickle as pkl

from load_csv_into_rows import load_csv, SAMPLE_CSV_FILE, DATA_DIR

# generating a sample question to a table
# with a complex key on 2 columns
# e.g. what was population in amsterdam in 2015?
question_template = "what was %s in %s in %s?"

key_columns = ['lau2_name', 'year']
answer_columns = ['internal_mig_immigration',
                  'international_mig_immigration',
                  'immigration_total',
                  'internal_mig_emigration',
                  'international_mig_emigration',
                  'emigration_total']


def generate_questions(file_name):
    header, rows = load_csv(SAMPLE_CSV_FILE)
    # row_strs = []
    data = []
    for i, row in enumerate(rows):
        row_str = ""
        keys = []
        # container to store tuples of question-answer pairs
        qa_keys = []
        qa_s = []
        # go over the row
        for j, cell in enumerate(row):
            if header[j].lower() in key_columns:
                keys.append(cell)
            elif header[j].lower() in answer_columns:
                qa_keys.append((header[j], cell))
            # row as a string:
            row_str = "%s %s %s" % (row_str, header[j], cell)
            row_str = row_str.lower().strip()
        for qa in qa_keys:
            question = question_template % (qa[0], keys[0], keys[1])
            answer = qa[1]
            qa_s.append((question.lower(), answer.lower()))
        data.append((row_str, qa_s))
    return data


def test_generate_questions(file_name=SAMPLE_CSV_FILE):
    # generate
    data = generate_questions(file_name)
    loc = DATA_DIR + file_name[:-4] + '_data.pkl'
    with open(loc, 'w') as f:
        pkl.dump(data, f)

    # print table stats
    nrows = len(data)
    nqas = len(data[0][1])
    print 'Table with %i rows' % nrows
    print 'Questions for each row: %i \n' % nqas

    # show sample
    sample_row = randrange(0, nrows)
    print 'Row #%i:' % sample_row
    print data[sample_row][0]
    sample_qa = randrange(0, nqas)
    print 'Sample QA:', data[sample_row][1][sample_qa]


if __name__ == '__main__':
    test_generate_questions()
