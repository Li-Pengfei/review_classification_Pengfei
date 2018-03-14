import string
import numpy as np
import pandas as pd
import nltk
import cPickle
import numpy as np
import itertools
from collections import Counter
from sklearn import preprocessing


def convert_label(annot):
    annot = np.char.lower(annot)

    uni_label = np.unique(annot).tolist()
    label = [uni_label.index(x) for x in annot]

    idx2label = {idx: word for idx, word in enumerate(uni_label)}

    return [label, Counter(annot.tolist()), uni_label]


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
    thefile = open('' + filename, 'w')
    for sentence in file:
        thefile.write("%s\n" % sentence)
    thefile.close()


def write_Label(content, listuple, csv_path, index):
    # creating pandas
    idx_comment = range(len(content))
    df_comment = pd.DataFrame({'id': idx_comment, 'comment': content})
    df_survey = pd.DataFrame(listuple, columns=['sentence', 'id', 'label'])
    df_all = pd.merge(df_comment, df_survey, on='id', how='inner')
    df_final = df_all.groupby(['id', 'comment'])['label'].apply(list).to_frame()

    df_final['label'] = df_final['label'].apply(lambda x: rule_all(x))

    # Write label to original CSV file
    with open(csv_path, 'r') as csvinput:
        reader = csv.reader(csvinput)
        all = []
        row = next(reader)
        if len(row) == 36:
            row.append('Class_Label')
        all.append(row)

        count = 0
        for i, row in enumerate(reader):
            if i in index and len(row) == 36:
                row.append(df_final['label'].values[count])
                count += 1
            all.append(row)
    with open(csv_path, 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        writer.writerows(all)

