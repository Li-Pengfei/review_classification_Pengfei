import numpy as np
import pandas as pd
import random
import cPickle
import os.path
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

currentpath = '/data1/shared_all/review_classification_Pengfei-master'
os.chdir(currentpath)


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


def load_data(data_source, y, k):

    x, sequence_length, vocabulary, vocabulary_inv_list = data_helpers.load_data(data_source)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    ddata = data_source[shuffle_indices]

    # train_len = int(len(x) * 0.9)
    # x_train = x[:train_len]
    # y_train = y[:train_len]
    # x_test = x[train_len:]
    # y_test = y[train_len:]
    # yn_test = yn[train_len:]
    # data_test = ddata[train_len:]

    embedding_file = './API/word2vec.p'
    if os.path.exists(embedding_file):
        embedding_weights = cPickle.load(open(embedding_file, "rb"))
    else:
        embedding_weights = data_helpers.load_bin_vec('/data1/shared_all/GoogleNews-vectors-negative300.bin', [new_vocab for new_vocab in vocabulary])
        cPickle.dump(embedding_weights, open(embedding_file, "wb"))

    # K-fold CV
    kf = KFold(n_splits=k)
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for train, test in kf.split(x):
        X_train.append(np.stack([np.stack([embedding_weights[vocabulary_inv[word]] for word in sentence]) for sentence in x[train]]))
        X_test.append(np.stack([np.stack([embedding_weights[vocabulary_inv[word]] for word in sentence]) for sentence in x[test]]))
        Y_train.append(y[train])
        Y_test.append(y[test])

    return X_train, Y_train, X_test, Y_test, vocabulary_inv, sequence_length


def top_n_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


xlsx_file = './data/HT Data.xlsx'
roi_list = ['HT Experience', 'Context', 'Content', 'Driver']
data, label = read_Surveycsv(xlsx_file, roi_list)

# Data Preparation
print("Load data...")
label_name = 'content'
k_fold = 5
if label_name is 'context':
    nb_classes = len(label[0][2])
    labels = label[0][2]
    idx2label = {idx: word for idx, word in enumerate(labels)}
    label_counter = label[0][1]
    print("Labels:", labels)
    print("Index to label:", idx2label)
    print("Label count:", label_counter)
    sub_label = np.asarray(label[0][0])

elif label_name is 'content':
    nb_classes = len(label[1][2])
    labels = label[1][2]
    idx2label = {idx: word for idx, word in enumerate(labels)}
    label_counter = label[1][1]
    print("Labels:", labels)
    print("Index to label:", idx2label)
    print("Label count:", label_counter)
    sub_label = np.asarray(label[1][0])
elif label_name is 'driver':
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
X_train, Y_train, X_test, Y_test, vocabulary_inv, sequence_length = load_data(data, sub_label, k_fold)

# Model Hyperparameters
embedding_dim = 300
vocabsize = len(vocabulary_inv)
dropout_prob = 0.5
batch_size = 30
num_epochs = 10
nb_filter = 100
hidden_dims = 100
filter_len = [1, 2, 3, 4]


# Build CNN
def build_model(ft=False):
<<<<<<< HEAD
    print("Building CNN...")
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
    print("hidden dim:", hidden_dims)
    # z1 = BatchNormalization()(z1)
    z1 = Activation('tanh')(z1)
    z1 = Dropout(dropout_prob)(z1)
    z1 = Dense(nb_classes)(z1)
    # z1 = BatchNormalization()(z1)
    model_output1 = Activation('softmax')(z1)
    cnn_model = Model(inputs=graph_input1, outputs=model_output1)

    return cnn_model
=======
	print("Building CNN...")
	if ft==False:
		graph_input1 = Input(shape=(sequence_length,embedding_dim))
		embed1 = Embedding(vocabsize, embedding_dim,input_length=sequence_length,name="embedding")(graph_input1)
	convs1 = []
	for fsz in filter_len:
		conv1 = Conv1D(filters=nb_filter,kernel_size=fsz,activation='relu')(graph_input1)
		pool1 = GlobalMaxPooling1D()(conv1)
		convs1.append(pool1)
	y1 = Concatenate()(convs1) if len(convs1) > 1 else convs1[0]
	z1 = Dropout(dropout_prob)(y1)
	z1 = Dense(nb_classes)(z1)
	z1 = BatchNormalization()(z1)
	model_output1 = Activation('softmax')(z1)
	cnn_model = Model(inputs=graph_input1, outputs=model_output1)

	return cnn_model
>>>>>>> cc07b9943384209eb8d1355377f08b9905d96af1


# Train CNN
print("Training CNN...")
pred, y_test = [], []
for i in range(k_fold):
<<<<<<< HEAD
    y_trains = to_categorical(Y_train[i], num_classes=nb_classes)
    y_tests = to_categorical(Y_test[i], num_classes=nb_classes)
    cnn_model = build_model(ft=False)
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train[i], y_trains, epochs=num_epochs, batch_size=batch_size, verbose=2)
    pred = pred + np.argmax(cnn_model.predict(X_test[i]), axis=1).tolist()
    y_test = y_test + Y_test[i].tolist()
    print('%sth fold finished' % str(i))
    K.clear_session()

cm = confusion_matrix(y_test, pred)
print('%s overall accuracy is : %0.2f' % (label_name, accuracy_score(y_test, pred) * 100))
=======
	y_trains = to_categorical(Y_train[i],num_classes=nb_classes)
	y_tests = to_categorical(Y_test[i],num_classes=nb_classes)
	cnn_model = build_model(ft=False)
	cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	cnn_model.fit(X_train[i], y_trains, epochs=num_epochs, batch_size=batch_size, verbose=2)	
	pred = pred + np.argmax(cnn_model.predict(X_test[i]),axis=1).tolist()
	y_test = y_test + Y_test[i].tolist()
	print('%sth fold finished'%str(i))
	K.clear_session()

cm = confusion_matrix(y_test,pred)
print('%s overall accuracy is : %0.2f'%(label_name,accuracy_score(y_test,pred)*100))
>>>>>>> cc07b9943384209eb8d1355377f08b9905d96af1

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=labels,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=labels, normalize=True,
                      title='Normalized confusion matrix')
plt.show()


# Output wrong predictions to csv file
csv_path = './pred_results/prediction_%s.csv' % label_name

true_label = [idx2label[i] for i in y_test]
pred_label = [idx2label[i] for i in pred]

df = pd.DataFrame({'HT_Experience': data,
                   'True_Label': true_label,
                   'True_Label_idx': y_test,
                   'Predicted_Label': pred_label,
                   'Predicted_Label_idx': pred})
df = df.sort_values(by=['True_Label_idx', 'Predicted_Label_idx'])
df.to_csv(csv_path, encoding='utf-8')
print('Prediction results written to: ', csv_path)


# Record top-n accuracy for "content"
# cnn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[top_n_accuracy])
# cnn_model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2)
# y_pred = [x.argsort()[-3:][::-1] for x in cnn_model.predict(x_test)]
# # survey_reader.write_Label(y_pred,labels,shuffle_index)
# thefile = open('./API_phase2/content.txt', 'w')
# for i,sentence in enumerate(data_test):
# 	pred_label = [new_labelname[z] for z in y_pred[i]]
# 	thefile.write("%s\t" % sentence.encode('ascii', 'ignore').decode('ascii'))
# 	thefile.write(" ")
# 	thefile.write("%s\t" % yn_test[i])
# 	thefile.write(" ")
# 	for j in pred_label:
# 		thefile.write("%s\t" % j)
# 		thefile.write("\n")
# thefile.close()

# prob = cnn_model.predict(x_test, verbose=2)
# prob1 = prob[:,:nb_classes[0]]
# prob2 = prob[:,nb_classes[0]:nb_classes[0]+nb_classes[1]]
# prob3 = prob[:,-nb_classes[2]:]
# y_test1 = y_test[:,:nb_classes[0]]
# y_test2 = y_test[:,nb_classes[0]:nb_classes[0]+nb_classes[1]]
# y_test3 = y_test[:,-nb_classes[2]:]
# print ('context mean AP: ', average_precision_score(y_test1,prob1))
# print ('content mean AP: ', average_precision_score(y_test2,prob2))
# print ('driver mean AP: ', average_precision_score(y_test3,prob3))
