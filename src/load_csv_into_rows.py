# -*- coding: utf-8 -*-
'''
author: svakulenko
18 Jul 2017
'''

import csv


DATA_DIR = '../data/'
SAMPLE_CSV_FILE = 'OOE_Wanderungen_Zeitreihe.csv'


def load_csv(file_name):
    header = None
    rows = []
    with open(DATA_DIR+file_name, 'rb') as f:
        reader = csv.reader(f, delimiter=";")
        for i, line in enumerate(reader):
            if i == 0:
                header = line
            else:
                rows.append(line)
    return header, rows


def table2rows(file_name):
    header, rows = load_csv(SAMPLE_CSV_FILE)
    row_strs = []
    for i, row in enumerate(rows):
        row_str = ""
        for j, cell in enumerate(row):
            row_str = "%s %s %s" % (row_str, header[j], cell)
        row_strs.append(row_str.lower().strip())
    return row_strs


def test_table2rows():
    print table2rows(SAMPLE_CSV_FILE)[:2]


def test_load_csv():
    header, rows = load_csv(SAMPLE_CSV_FILE)
    print header
    print rows[0]


if __name__ == '__main__':
    test_table2rows()
