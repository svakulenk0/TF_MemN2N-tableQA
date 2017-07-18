# -*- coding: utf-8 -*-
'''
author: svakulenko
18 Jul 2017
'''

from load_csv_into_rows import load_csv, SAMPLE_CSV_FILE

DATA_DIR = './data/'
SAMPLE_CSV_FILE = 'OOE_Wanderungen_Zeitreihe.csv'

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
    qa_s = []
    for i, row in enumerate(rows):
        row_str = ""
        keys = []
        # container to store tuples of question-answer pairs
        qa_keys = []
        # go over the row
        for j, cell in enumerate(row):
            if header[j].lower() in key_columns:
                keys.append(cell)
            elif header[j].lower() in answer_columns:
                qa_keys.append((header[j], cell))
            # row as a string:
            # row_str = "%s %s %s" % (row_str, header[j], cell)
        # row_strs.append(row_str.lower().strip())
        for qa in qa_keys:
            question = question_template % (qa[0], keys[0], keys[1])
            answer = qa[1]
            qa_s.append((question.lower(), answer.lower()))
    return qa_s


def test_generate_questions():
    print generate_questions(SAMPLE_CSV_FILE)[:2]


if __name__ == '__main__':
    test_generate_questions()
