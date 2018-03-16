import numpy as np
import pandas as pd
import random
import os.path

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

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
from keras import metrics
from keras.constraints import maxnorm

from survey_reader import *
from w2v import train_word2vec
import data_helpers


w2v_path = './API/GoogleNews-vectors-negative300.bin'


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def load_data_CV(data_source, y, k):

    x, sequence_length, vocabulary, vocabulary_inv_list = data_helpers.load_data(data_source)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    ddata = data_source[shuffle_indices]

    # load word embeddings
    print("Loading word embeddings...")
    embedding_weights = data_helpers.load_bin_vec(w2v_path, [new_vocab for new_vocab in vocabulary])


    # K-fold CV
    kf = KFold(n_splits=k)
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for train, test in kf.split(x):
        X_train.append(np.stack([np.stack([embedding_weights[vocabulary_inv[word]] for word in sentence]) for sentence in x[train]]))
        X_test.append(np.stack([np.stack([embedding_weights[vocabulary_inv[word]] for word in sentence]) for sentence in x[test]]))
        Y_train.append(y[train])
        Y_test.append(y[test])

    return X_train, Y_train, X_test, Y_test, vocabulary_inv, sequence_length, ddata


def load_train_data(data_source, y):
    x, sequence_length, vocabulary, vocabulary_inv_list = data_helpers.load_data(data_source)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    ddata = data_source[shuffle_indices]

    # load word embeddings
    print("Loading word embeddings...")
    embedding_weights = data_helpers.load_bin_vec(w2v_path,[new_vocab for new_vocab in vocabulary])

    X_train = np.stack([np.stack([embedding_weights[vocabulary_inv[word]] for word in sentence]) for sentence in x])
    Y_train = y

    return X_train, Y_train, vocabulary_inv, sequence_length, ddata

def load_test_data(data_source, sequence_length):
    x, sequence_length, vocabulary, vocabulary_inv_list = data_helpers.load_data(data_source, sequence_length)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

    # load word embeddings
    print("Loading word embeddings...")
    embedding_weights = data_helpers.load_bin_vec(w2v_path,[new_vocab for new_vocab in vocabulary])

    X_test = np.stack([np.stack([embedding_weights[vocabulary_inv[word]] for word in sentence]) for sentence in x])

    return X_test

def top_n_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)




def train_CV(train_file, category, result_file, k_fold=5):

    # Data Preparation
    print("Load data...")
    roi_list = ['HT Experience', 'Context', 'Content', 'Driver']
    data, label = read_Surveycsv(train_file, roi_list)

    if category is 'context':
        nb_classes = len(label[0][2])
        labels = label[0][2]
        idx2label = {idx: word for idx, word in enumerate(labels)}
        label_counter = label[0][1]
        print("Labels:", labels)
        print("Index to label:", idx2label)
        print("Label count:", label_counter)
        sub_label = np.asarray(label[0][0])
    elif category is 'content':
        nb_classes = len(label[1][2])
        labels = label[1][2]
        idx2label = {idx: word for idx, word in enumerate(labels)}
        label_counter = label[1][1]
        print("Labels:", labels)
        print("Index to label:", idx2label)
        print("Label count:", label_counter)
        sub_label = np.asarray(label[1][0])
    elif category is 'driver':
        nb_classes = len(label[2][2])
        labels = label[2][2]
        idx2label = {idx: word for idx, word in enumerate(labels)}
        label_counter = label[2][1]
        print("Labels:", labels)
        print("Index to label:", idx2label)
        print("Label count:", label_counter)
        sub_label = np.asarray(label[2][0])
    else:
        class_name = ['context', 'content', 'driver']
        nb_classes = [len(label[0][1]), len(label[1][1]), len(label[2][1])]
        sub_label1 = to_categorical(label[0][0], num_classes=nb_classes[0])
        sub_label2 = to_categorical(label[1][0], num_classes=nb_classes[1])
        sub_label3 = to_categorical(label[2][0], num_classes=nb_classes[2])
        sub_label = np.concatenate((sub_label1, sub_label2, sub_label3), axis=1)
    X_train, Y_train, X_test, Y_test, vocabulary_inv, sequence_length, original_data = load_data_CV(data, sub_label, k_fold)

    # Function for building CNN
    def build_model(ft=False):
        # CNN Model Hyperparameters
        embedding_dim = 300
        vocabsize = len(vocabulary_inv)
        dropout_prob = 0.5
        nb_filter = 100
        hidden_dims = 100
        filter_len = [1, 2, 3, 4]

        if ft == False:
            graph_input1 = Input(shape=(sequence_length, embedding_dim))
            embed1 = Embedding(vocabsize, embedding_dim, input_length=sequence_length, name="embedding")(graph_input1)
        convs1 = []
        for fsz in filter_len:
            conv1 = Conv1D(filters=nb_filter, kernel_size=fsz, activation='relu')(graph_input1)
            pool1 = GlobalMaxPooling1D()(conv1)
            convs1.append(pool1)

        y1 = Concatenate()(convs1) if len(convs1) > 1 else convs1[0]
        z1 = Dropout(dropout_prob)(y1)
        z1 = Dense(hidden_dims, kernel_constraint=maxnorm(2))(z1)
        z1 = BatchNormalization()(z1)
        z1 = Activation('tanh')(z1)
        z1 = Dropout(dropout_prob)(z1)
        z1 = Dense(nb_classes)(z1)
        z1 = BatchNormalization()(z1)
        model_output1 = Activation('softmax')(z1)
        cnn_model = Model(inputs=graph_input1, outputs=model_output1)

        return cnn_model

    # Train CNN
    batch_size = 30
    num_epochs = 10
    pred, y_test = [], []
    pred_prob = []
    for i in range(k_fold):
        y_trains = to_categorical(Y_train[i], num_classes=nb_classes)
        y_tests = to_categorical(Y_test[i], num_classes=nb_classes)
        print("Building CNN...")
        cnn_model = build_model(ft=False)
        cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Training CNN...")
        cnn_model.fit(X_train[i], y_trains, epochs=num_epochs, batch_size=batch_size, verbose=2)
        # predict testing data
        pred = pred + np.argmax(cnn_model.predict(X_test[i]), axis=1).tolist()
        pred_prob = pred_prob + cnn_model.predict(X_test[i]).tolist()
        y_test = y_test + Y_test[i].tolist()
        print('%sth fold finished' % str(i))
        K.clear_session()

    print('%s overall accuracy is : %0.2f' % (category, accuracy_score(y_test, pred) * 100))

    # Plot non-normalized confusion matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure()
    plot_confusion_matrix(cm, classes=labels, title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=labels, normalize=True, title='Normalized confusion matrix')
    plt.show()

    # Output prediction results to csv file
    true_label = [idx2label[i] for i in y_test]
    pred_label = [idx2label[i] for i in pred]
    # sorted prediction probability
    pred_label_prob = [[(idx2label[i], round(prob * 100, 2))
                        for i, prob in sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)]
                       for probabilities in pred_prob]
    # unsorted prediction probability
    # pred_label_prob = [[(idx2label[i], round(prob * 100, 2)) for i, prob in enumerate(probabilities)] for probabilities in pred_prob]

    df = pd.DataFrame({'HT_Experience': original_data,
                       'True_Label': true_label,
                       'True_Label_idx': y_test,
                       'Predicted_Label': pred_label,
                       'Predicted_Label_idx': pred,
                       'Pred_Label_prob': pred_label_prob})
    df = df.sort_values(by=['True_Label_idx', 'Predicted_Label_idx'])
    df.to_csv(result_file, encoding='utf-8')
    print('Prediction results written to: ', result_file)


def train(train_file, category):
    # Data Preparation
    print("Load data...")
    roi_list = ['HT Experience', 'Context', 'Content', 'Driver']
    data, label = read_Surveycsv(train_file, roi_list)

    if category is 'context':
        nb_classes = len(label[0][2])
        labels = label[0][2]
        idx2label = {idx: word for idx, word in enumerate(labels)}
        label_counter = label[0][1]
        print("Labels:", labels)
        print("Index to label:", idx2label)
        print("Label count:", label_counter)
        sub_label = np.asarray(label[0][0])
    elif category is 'content':
        nb_classes = len(label[1][2])
        labels = label[1][2]
        idx2label = {idx: word for idx, word in enumerate(labels)}
        label_counter = label[1][1]
        print("Labels:", labels)
        print("Index to label:", idx2label)
        print("Label count:", label_counter)
        sub_label = np.asarray(label[1][0])
    elif category is 'driver':
        nb_classes = len(label[2][2])
        labels = label[2][2]
        idx2label = {idx: word for idx, word in enumerate(labels)}
        label_counter = label[2][1]
        print("Labels:", labels)
        print("Index to label:", idx2label)
        print("Label count:", label_counter)
        sub_label = np.asarray(label[2][0])
    else:
        class_name = ['context', 'content', 'driver']
        nb_classes = [len(label[0][1]), len(label[1][1]), len(label[2][1])]
        sub_label1 = to_categorical(label[0][0], num_classes=nb_classes[0])
        sub_label2 = to_categorical(label[1][0], num_classes=nb_classes[1])
        sub_label3 = to_categorical(label[2][0], num_classes=nb_classes[2])
        sub_label = np.concatenate((sub_label1, sub_label2, sub_label3), axis=1)
    X_train, Y_train, vocabulary_inv, sequence_length, original_data = load_train_data(data, sub_label)

    # Function for building CNN
    def build_model(ft=False):
        # CNN Model Hyperparameters
        embedding_dim = 300
        vocabsize = len(vocabulary_inv)
        dropout_prob = 0.5
        nb_filter = 100
        hidden_dims = 100
        filter_len = [1, 2, 3, 4]

        if ft == False:
            graph_input1 = Input(shape=(sequence_length, embedding_dim))
            embed1 = Embedding(vocabsize, embedding_dim, input_length=sequence_length, name="embedding")(graph_input1)
        convs1 = []
        for fsz in filter_len:
            conv1 = Conv1D(filters=nb_filter, kernel_size=fsz, activation='relu')(graph_input1)
            pool1 = GlobalMaxPooling1D()(conv1)
            convs1.append(pool1)

        y1 = Concatenate()(convs1) if len(convs1) > 1 else convs1[0]
        z1 = Dropout(dropout_prob)(y1)
        z1 = Dense(hidden_dims, kernel_constraint=maxnorm(2))(z1)
        z1 = BatchNormalization()(z1)
        z1 = Activation('tanh')(z1)
        z1 = Dropout(dropout_prob)(z1)
        z1 = Dense(nb_classes)(z1)
        z1 = BatchNormalization()(z1)
        model_output1 = Activation('softmax')(z1)
        cnn_model = Model(inputs=graph_input1, outputs=model_output1)

        return cnn_model

    # Train CNN
    batch_size = 30
    num_epochs = 10

    y_trains = to_categorical(Y_train, num_classes=nb_classes)
    print("Building CNN...")
    cnn_model = build_model(ft=False)
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Training CNN...")
    cnn_model.fit(X_train, y_trains, epochs=num_epochs, batch_size=batch_size, verbose=2)

    return [cnn_model, idx2label, sequence_length]

def predict(test_file, cnn_model, result_file):
    trained_model = cnn_model[0]
    idx2label = cnn_model[1]
    sequence_length = cnn_model[2]

    data = pd.read_excel(test_file, usecols=['HT Experience']).values.tolist()
    data = [sent[0] for sent in data]

    X_test = load_test_data(data, sequence_length)

    pred, pred_prob = [], []
    # predict testing data
    pred = pred + np.argmax(trained_model.predict(X_test), axis=1).tolist()
    pred_prob = pred_prob + trained_model.predict(X_test).tolist()

    # Output prediction results to csv file
    pred_label = [idx2label[i] for i in pred]
    # sorted prediction probability
    pred_label_prob = [[(idx2label[i], round(prob * 100, 2))
                        for i, prob in sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)]
                       for probabilities in pred_prob]
    # unsorted prediction probability
    # pred_label_prob = [[(idx2label[i], round(prob * 100, 2)) for i, prob in enumerate(probabilities)] for probabilities in pred_prob]

    df = pd.DataFrame({'HT_Experience': data,
                       'Predicted_Label': pred_label,
                       'Predicted_Label_idx': pred,
                       'Pred_Label_prob': pred_label_prob})
    # df = df.sort_values(by=['Predicted_Label_idx'])
    df.to_csv(result_file, encoding='utf-8')
    print('Prediction results written to: ', result_file)





if __name__ == '__main__':
    train_CV('./data/HT Data.xlsx', 'content', './pred_results/prediction_content.csv')
    # cnn_model = train('./data/HT Data.xlsx', 'content')
    # predict('./data/HT Data_test.xlsx', cnn_model, './pred_results/prediction_test.csv')
