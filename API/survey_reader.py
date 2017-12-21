import string
import numpy as np
import pandas as pd
import nltk
import cPickle
import numpy as np
import data_helper
import itertools
from collections import Counter
from sklearn import preprocessing


def convert_label(annot):
    annot = np.char.lower(annot)

    uni_label = np.unique(annot).tolist()
    label = [uni_label.index(x) for x in annot]

    label2idx = {word: idx for idx, word in enumerate(uni_label)}

    # uniq_label = []
    # for u in uni_label:
    #     uniq_label.append(annot.tolist().count(u))

    return [label, Counter(annot.tolist()), label2idx]


def read_Surveycsv(filepath, roilist):

    df = pd.read_excel(filepath, usecols=roilist)
    # print(df.describe())
    all_data = np.asarray(df.values.tolist())
    data = all_data[:, 0]
    context_label = convert_label(all_data[:, 1])
    content_label = convert_label(all_data[:, 2])
    driver_label = convert_label(all_data[:, 3])

    return [data, [context_label, content_label, driver_label]]


def fileWriter(filename, file):
    """
    author: Pengfei
    date: 07/06/2017
    """
    thefile = open('C:/Users/pli006/Documents/Sourcetree/review_cluster/data/' + filename, 'w')
    for sentence in file:
        thefile.write("%s\n" % sentence)
    thefile.close()


if __name__ == '__main__':

    xlsx_file = '/home/dongzhe/Documents/review_cluster/data/HT Data.xlsx'
    roi_list = ['HT Experience', 'Context', 'Content', 'Driver']
    data, label = read_Surveycsv(xlsx_file, roi_list)
