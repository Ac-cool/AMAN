from __future__ import print_function
import json
import os
import re
import time
import numpy as np
import tensorflow as tf
import pandas as pd

import codecs
#from gensim.models import Word2Vec
import keras
import pickle
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
from keras.layers.recurrent import GRU,LSTM
from keras.backend.tensorflow_backend import set_session
from keras.engine.topology import Layer

from keras.utils import multi_gpu_model
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.activations import softmax





import numpy as np
import csv, datetime, time, json
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.utils.training_utils import multi_gpu_model

#import codecs
# Initialize global variables
KERAS_DATASETS_DIR = expanduser('/home/liangdi/quora/fubao/baseline_two_our_model/datasets/')
QUESTION_PAIRS_FILE_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
QUESTION_PAIRS_FILE = 'questionWithAnswer.txt'
GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
GLOVE_ZIP_FILE = 'glove.840B.300d.zip'
GLOVE_FILE = 'glove.840B.300d.txt'
Q1_TRAINING_DATA_FILE = 'q1_train.npy'
Q2_TRAINING_DATA_FILE = 'q2_train.npy'
LABEL_TRAINING_DATA_FILE = 'label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = 'nb_words.json'
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 300
MODEL_WEIGHTS_FILE = 'question_pairs_weights.check'
VALIDATION_SPLIT = 0.03
TEST_SPLIT = 0.2
RNG_SEED = 13371447
NB_EPOCHS = 24
DROPOUT = 0.05
BATCH_SIZE = 256
OPTIMIZER = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)

# If the dataset, embedding matrix and word count exist in the local directory
if exists(Q1_TRAINING_DATA_FILE) and exists(Q2_TRAINING_DATA_FILE) and exists(LABEL_TRAINING_DATA_FILE) and exists(NB_WORDS_DATA_FILE) and exists(WORD_EMBEDDING_MATRIX_FILE):
    # Then load them
    q1_data = np.load(open(Q1_TRAINING_DATA_FILE, 'rb'))
    q2_data = np.load(open(Q2_TRAINING_DATA_FILE, 'rb'))
    labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))
    word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
    with open(NB_WORDS_DATA_FILE, 'r') as f:
        nb_words = json.load(f)['nb_words']
else:
    # Else download and extract questions pairs data
    #if not exists(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE):
    #    get_file(QUESTION_PAIRS_FILE, QUESTION_PAIRS_FILE_URL)

    #print("Processing", QUESTION_PAIRS_FILE)

    question1 = []
    question2 = []
    answers= []
    answers2=[]
    is_duplicate = []
    nn =0
    with codecs.open("/home/liangdi/quora/fubao/all_all_two_2W.txt","r","utf8") as csvfile:
        reader = csvfile.read()
        reader = reader.split("\n##########\n")
        print("lines_num:",len(reader))
        for row in reader:
            ll = row.split("\t*###*\t")
            if len(ll)<5:
                print(row)
                continue
            question1.append(ll[0])
            question2.append(ll[1])
            answers.append(ll[2])
            answers2.append(ll[3])
            is_duplicate.append(ll[4])
            try :
                dddddd = int(ll[4])
            except:
                print(row)
    """
    with codecs.open(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE,"r","utf8") as csvfile:
        reader = csvfile.read()
        reader = reader.split("\n")
        print("train_count",len(reader))
        for row in reader:
            ll = row.split("\t*###*\t")
            if len(ll)<5:
                print(row)
                continue
            question1.append(ll[0])
            question2.append(ll[1])
            answers.append(ll[2])
            answers2.append(ll[3])
            is_duplicate.append(ll[4])
            try :
                dddddd = int(ll[4])
            except:
                print(row)
    """
    print('Question pairs: %d' % len(question1))
    print(nn)
    # Build tokenized word index
    questions = question1 + question2+answers+answers2
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,lower=True)
    #filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
    tokenizer.fit_on_texts(questions)
    question1_word_sequences = tokenizer.texts_to_sequences(question1)
    question2_word_sequences = tokenizer.texts_to_sequences(question2)
    answers_word_sequences = tokenizer.texts_to_sequences(answers)
    answers2_word_sequences = tokenizer.texts_to_sequences(answers2)
    word_index = tokenizer.word_index

    print("Words in index: %d" % len(word_index))

    # Download and process GloVe embeddings
    #if not exists(KERAS_DATASETS_DIR + GLOVE_ZIP_FILE):
    #    zipfile = ZipFile(get_file(GLOVE_ZIP_FILE, GLOVE_ZIP_FILE_URL))
    #    zipfile.extract(GLOVE_FILE, path=KERAS_DATASETS_DIR)

    print("Processing", GLOVE_FILE)

    embeddings_index = {}
    with codecs.open(KERAS_DATASETS_DIR + GLOVE_FILE,"r","utf8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings: %d' % len(embeddings_index))

    # Prepare word embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    ss  =""
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
        else:
            ss+=word+"\n"
    with codecs.open("./null_words","w","utf8")as f:
        f.write(ss)
    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

    # Prepare training data tensors
    q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    answer_data = pad_sequences(answers_word_sequences, maxlen=100)
    answer2_data = pad_sequences(answers2_word_sequences, maxlen=100)
    labels = np.array(is_duplicate, dtype=int)
    print('Shape of question1 data tensor:', q1_data.shape)
    print('Shape of question2 data tensor:', q2_data.shape)
    print('Shape of label tensor:', labels.shape)

    # Persist training and configuration data to files
    #np.save(open(Q1_TRAINING_DATA_FILE, 'wb'), q1_data)
    #np.save(open(Q2_TRAINING_DATA_FILE, 'wb'), q2_data)
    #np.save(open(LABEL_TRAINING_DATA_FILE, 'wb'), labels)
    #np.save(open(WORD_EMBEDDING_MATRIX_FILE, 'wb'), word_embedding_matrix)
    #with open(NB_WORDS_DATA_FILE, 'w') as f:
    #    json.dump({'nb_words': nb_words}, f)

# Partition the dataset into train and test sets
indices = np.arange(q1_data.shape[0])
np.random.shuffle(indices)

q1_data = q1_data[indices]
q2_data = q2_data[indices]
answer = answer_data[indices]
answer2 = answer2_data[indices]
labels = labels[indices]

Q1_test = q1_data[:10000]
Q2_test = q2_data[:10000]
answer_test = answer[:10000]
answer2_test = answer2[:10000]
y_test = labels[:10000]

Q1_dev = q1_data[10000:20000]
Q2_dev = q2_data[10000:20000]
answer_dev= answer[10000:20000]
answer2_dev= answer2[10000:20000]
y_dev = labels[10000:20000]

Q1_train = q1_data[20000:]
Q2_train = q2_data[20000:]
answer_train = answer[20000:]
answer2_train = answer2[20000:]
y_train = labels[20000:]

# Define the model
'''
question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

q1 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question1)
q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1)
q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q1)

q2 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question2)
q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2)
q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q2)

merged = concatenate([q1,q2])
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
'''

def aggregate(input_1, input_2,A_encoded,A2_encoded, num_dense=300, dropout_rate=0.2):
    feat1 = concatenate([GlobalAvgPool1D()(input_1), GlobalMaxPool1D()(input_1)])
    feat2 = concatenate([GlobalAvgPool1D()(input_2), GlobalMaxPool1D()(input_2)])
    feat3 = concatenate([GlobalAvgPool1D()(A_encoded), GlobalMaxPool1D()(A_encoded)])
    feat4 = concatenate([GlobalAvgPool1D()(A2_encoded), GlobalMaxPool1D()(A2_encoded)])    

    x = concatenate([feat1, feat2,feat3,feat4])
    x = BatchNormalization()(x)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x
def aggregate2(input_1, input_2, num_dense=300, dropout_rate=0.2):
    feat1 = concatenate([GlobalAvgPool1D()(input_1), GlobalMaxPool1D()(input_1)])
    feat2 = concatenate([GlobalAvgPool1D()(input_2), GlobalMaxPool1D()(input_2)])
    #feat3 = concatenate([GlobalAvgPool1D()(A_encoded), GlobalMaxPool1D()(A_encoded)])

    x = concatenate([feat1, feat2])
    x = BatchNormalization()(x)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x


def subtract(input_1, input_2):
    minus_input_2 = Lambda(lambda x: -x)(input_2)
    return add([input_1, minus_input_2])

def align(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

def align2(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

def align_A(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

def align_B(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
Answer = Input(shape=(100,))
Answer2 = Input(shape=(100,))

q1 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question1)
q2 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question2)
A = Embedding(nb_words + 1,
                 EMBEDDING_DIM,
                 weights=[word_embedding_matrix],
                 input_length=100,
                 trainable=False)(Answer)
A2 = Embedding(nb_words + 1,
                 EMBEDDING_DIM,
                 weights=[word_embedding_matrix],
                 input_length=100,
                 trainable=False)(Answer2)


#e1_aligned, e2_aligned = align(q1, q2)
#q1 = concatenate([q1,e2_aligned])
#q2 = concatenate([q2,e1_aligned])

Encoder = Bidirectional(LSTM(units=300, return_sequences=True))
q1_encoded = Dropout(DROPOUT)(Encoder(q1))
q2_encoded = Dropout(DROPOUT)(Encoder(q2))

Encoder_A = Bidirectional(LSTM(units=300, return_sequences=True))
A_encoded = Dropout(DROPOUT)(Encoder_A(A))
A2_encoded = Dropout(DROPOUT)(Encoder_A(A2))

q1_aligned, q2_aligned = align(q1_encoded, q2_encoded)
A1_aligned, A2_aligned = align2(A_encoded, A2_encoded)
q1_A_aligned, A_q1_aligned = align_A(A_encoded,q1_encoded)
q2_A_aligned, A_q2_aligned = align_B(A2_encoded,q1_encoded)

#q1_combined = concatenate([q1_encoded, q2_aligned, subtract(q1_encoded, q2_aligned), multiply([q1_encoded, q2_aligned]),q1_A_aligned])
#q2_combined = concatenate([q2_encoded, q1_aligned, subtract(q2_encoded, q1_aligned), multiply([q2_encoded, q1_aligned]),q2_A_aligned])
#A1_combined = concatenate([A_encoded, A2_aligned, subtract(A_encoded, A2_aligned), multiply([A_encoded, A2_aligned]),A_q1_aligned])
#A2_combined = concatenate([A2_encoded, A1_aligned, subtract(A2_encoded, A1_aligned), multiply([A2_encoded, A1_aligned]),A_q2_aligned])
q1_combined = concatenate([q1_encoded, q2_aligned, q1_A_aligned])
q2_combined = concatenate([q2_encoded, q1_aligned, q2_A_aligned])
A1_combined = concatenate([A_encoded, A2_aligned, A_q1_aligned])
A2_combined = concatenate([A2_encoded, A1_aligned, A_q2_aligned])

q1_combined = Dropout(DROPOUT)(q1_combined)
q2_combined = Dropout(DROPOUT)(q2_combined)
A1_combined = Dropout(DROPOUT)(A1_combined)
A2_combined = Dropout(DROPOUT)(A2_combined)

compare = Bidirectional(LSTM(300, return_sequences=True))
q1_compare = Dropout(DROPOUT)(compare(q1_combined))
q2_compare = Dropout(DROPOUT)(compare(q2_combined))
 
compare2 = Bidirectional(LSTM(300, return_sequences=True))
A1_compare = Dropout(DROPOUT)(compare2(A1_combined))
A2_compare = Dropout(DROPOUT)(compare2(A2_combined))
q_new = concatenate([q1_compare, q2_compare, subtract(q1_compare, q2_compare), multiply([q1_compare, q2_compare])])
a_new = concatenate([A1_compare, A2_compare, subtract(A1_compare, A2_compare), multiply([A1_compare, A2_compare])])

x = aggregate2(q_new,a_new)
#compare = Bidirectional(LSTM(300, return_sequences=True))
#q1_compare = Dropout(DROPOUT)(compare(q1_combined))
#q2_compare = Dropout(DROPOUT)(compare(q2_combined))

#compare2 = Bidirectional(LSTM(300, return_sequences=True))
#A1_compare = Dropout(DROPOUT)(compare2(A1_combined))
#A2_compare = Dropout(DROPOUT)(compare2(A2_combined))

#x = aggregate(q1_compare, q2_compare,A1_compare,A2_compare)
#x = aggregate(q1_combined, q2_combined,A1_combined,A2_combined)

is_duplicate = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[question1,question2,Answer,Answer2], outputs=is_duplicate)
#model = multi_gpu_model(model, gpus=6)
model.summary()
#model = multi_gpu_model(model, gpus=4)
model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# Train the model, checkpointing weights with best validation accuracy
print("Starting training at", datetime.datetime.now())
t0 = time.time()
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True,save_weights_only=True)]
history = model.fit([Q1_train, Q2_train,answer_train,answer2_train],
                    y_train,
                    epochs=NB_EPOCHS,
                    validation_data=([Q1_dev,Q2_dev,answer_dev,answer2_dev],y_dev),
                    verbose=2,
                    batch_size=BATCH_SIZE,
                    class_weight = 'auto',
                    callbacks=callbacks)
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

# Print best validation accuracy and epoch
max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
print('Maximum validation accuracy = {0:.4f} (epoch {1:d})'.format(max_val_acc, idx+1))

# Evaluate the model with best validation accuracy on the test partition
model.load_weights(MODEL_WEIGHTS_FILE)
loss, accuracy = model.evaluate([Q1_test, Q2_test,answer_test,answer2_test], y_test, verbose=0)
print('Test loss = {0:.4f}, test accuracy = {1:.4f}'.format(loss, accuracy))
