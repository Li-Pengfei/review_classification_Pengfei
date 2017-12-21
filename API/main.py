import numpy as np
import random
import cPickle
import os.path
import math
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import tensorflow as tf
from keras.utils import *
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Embedding, Input, Conv1D
from keras.layers.merge import Maximum, Concatenate, Average
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.constraints import maxnorm

from survey_reader import *
from w2v import train_word2vec
import data_helpers

currentpath = '/data1/shared_all/review_classification_Pengfei/'
os.chdir(currentpath)

np.random.seed(666)  # for reproducibility


def load_data(data_source, y, y_int):

    x, sequence_length, vocabulary, vocabulary_inv_list = data_helpers.load_data(data_source)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}  # index to word

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    y_int = y_int[shuffle_indices]

    train_len = int(len(x) * 0.8)
    x_train = x[:train_len]
    y_train = y[:train_len]
    y_int_train = y_int[:train_len]

    x_test = x[train_len:]
    y_test = y[train_len:]
    y_int_test = y_int[train_len:]

    embedding_file = './API/word2vec.p'
    if os.path.exists(embedding_file):
        embedding_weights = cPickle.load(open(embedding_file, "rb"))
    else:
        embedding_weights = data_helpers.load_bin_vec('/data1/shared_all/GoogleNews-vectors-negative300.bin', [new_vocab for new_vocab in vocabulary])
        cPickle.dump(embedding_weights, open(embedding_file, "wb"))

    x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
    x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])

    return x_train, y_train, x_test, y_test, vocabulary_inv, sequence_length, y_int_test


xlsx_file = './data/HT Data.xlsx'
roi_list = ['HT Experience', 'Context', 'Content', 'Driver']
data, label = read_Surveycsv(xlsx_file, roi_list)

# Data Preparation
print("Load data...")
label_name = 'content'
if label_name is 'context':
    label_counter = label[0][1].most_common()
    label2idx = label[0][2]
    print(label_counter)
    print(label2idx)
    nb_classes = len(label_counter)
    label_int = np.array(label[0][0])
    sub_label = to_categorical(label_int, num_classes=nb_classes)
elif label_name is 'content':
    label_counter = label[1][1].most_common()
    label2idx = label[1][2]
    print(label_counter)
    print(label2idx)
    nb_classes = len(label_counter)
    label_int = np.array(label[1][0])
    sub_label = to_categorical(label_int, num_classes=nb_classes)
else:
    label_counter = label[2][1].most_common()
    label2idx = label[2][2]
    print(label_counter)
    print(label2idx)
    nb_classes = len(label_counter)
    label_int = np.array(label[2][0])
    sub_label = to_categorical(label_int, num_classes=nb_classes)

x_train, y_train, x_test, y_test, vocabulary_inv, sequence_length, y_int_test = load_data(data, sub_label, label_int)
print("sequence length:", sequence_length)

# Model Hyperparameters
embedding_dim = 300
vocabsize = len(vocabulary_inv)
dropout_prob = 0.3
batch_size = 50
num_epochs = 20
nb_filter = 50
filter_len = [1, 2, 3, 4]
# hidden_dims = int(math.floor((len(filter_len) * nb_filter + nb_classes) / 3))
hidden_dims = 100


# Build CNN
print("Building CNN...")
graph_input1 = Input(shape=(sequence_length, embedding_dim))
# embed1 = Embedding(vocabsize, embedding_dim, input_length=sequence_length, name="embedding")(graph_input1)
convs1 = []
for fsz in filter_len:
    conv1 = Conv1D(filters=nb_filter, kernel_size=fsz, activation='tanh', padding='same',)(graph_input1)
    pool1 = GlobalMaxPooling1D()(conv1)
    convs1.append(pool1)
# y1 = Concatenate()(convs1) if len(convs1) > 1 else convs1[0]
y1 = Concatenate()(convs1)
z1 = Dropout(dropout_prob)(y1)

z1 = Dense(hidden_dims, kernel_constraint=maxnorm(2))(z1)
print("hidden dim:", hidden_dims)
# z1 = BatchNormalization()(z1)
z1 = Activation('tanh')(z1)
z1 = Dropout(dropout_prob)(z1)

z1 = Dense(nb_classes)(z1)
# z1 = BatchNormalization()(z1)
model_output1 = Activation('softmax')(z1)
cnn_model = Model(inputs=graph_input1, outputs=model_output1)
cnn_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train CNN
print("Training CNN...")
acc_list = []
max_acc = 0
max_acc_epoch = 0

for i in range(1, num_epochs + 1):
    print("\nTraining for epoch %d / %d" % (i, num_epochs))
    cnn_model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=2)
    # loss1, acc = cnn_model.evaluate(x_test, y_test, verbose=2)
    pred_test_prob = cnn_model.predict(x_test, verbose=False)
    # predict the class label:
    if pred_test_prob.shape[-1] > 1:
        pred_test = pred_test_prob.argmax(axis=-1)
    else:
        pred_test = (pred_test_prob > 0.5).astype('int32')

    if label_name == 'content':
        pred_test_top3 = np.argsort(pred_test_prob, axis=1)[:, ::-1][:, :3]
        print pred_test_top3.shape

    confusion_mtx = confusion_matrix(y_int_test, pred_test)
    print("confusion_matrix:")
    print(confusion_mtx)
    sum_ground_truth = np.sum(confusion_mtx, axis=1, dtype=np.float)
    confusion_mtx_recall = confusion_mtx / sum_ground_truth[:, None]
    confusion_mtx_recall = np.around(confusion_mtx_recall, decimals=3)
    print("confusion_matrix_recall:")
    print(confusion_mtx_recall)

    acc = np.sum(pred_test == y_int_test) / float(len(y_int_test))
    acc_list.append(acc * 100)

    if max_acc < acc:
        max_acc = acc
        max_acc_epoch = i

    # find accuracy (recall) for each class label
    # consider top 3 predicted label as correct, find accuracy (recall) for each class label
    for item in label_counter:
        label_word = item[0]
        targetLabel = label2idx[label_word]
        prec = data_helpers.getPrecision(pred_test, y_int_test, targetLabel)
        rec = data_helpers.getPrecision(y_int_test, pred_test, targetLabel)
        f1 = 0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

        # rec_top3 = data_helpers.getRecall_top3(y_int_test, pred_test_top3, targetLabel)
        print("Accuracy for '%s': %.4f" % (label_word, rec))

    print("'%s' overall accuracy: %.4f (max: %.4f at epoch %d)" % (label_name, acc * 100, max_acc * 100, max_acc_epoch))


# plot accuracy  for every epoch
fig = plt.figure(1, figsize=(15, 10))
ax = fig.add_subplot(111)
ax.plot(range(num_epochs), acc_list, 'b', label='Accuracy(%)')
plt.axis([0, num_epochs, 0, 100])

plt.xticks(np.arange(0, num_epochs + 1, 5.0))
plt.yticks(np.arange(0, 101, 5.0))
plt.ylabel('accuracy(%) ')
plt.xlabel('No. of epoch')
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.subplots_adjust(right=0.8)
plt.show()
