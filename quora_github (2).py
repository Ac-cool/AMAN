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
from test import read_snli




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
KERAS_DATASETS_DIR = expanduser('./datasets/')
QUESTION_PAIRS_FILE_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
QUESTION_PAIRS_FILE = 'quora_duplicate_questions.tsv'
GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
GLOVE_ZIP_FILE = 'glove.840B.300d.zip'
GLOVE_FILE = 'glove.840B.300d.txt'
Q1_TRAINING_DATA_FILE = 'q1_train.npy'
Q2_TRAINING_DATA_FILE = 'q2_train.npy'
LABEL_TRAINING_DATA_FILE = 'label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = 'nb_words.json'
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH =50
EMBEDDING_DIM = 300
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 24
DROPOUT = 0.1
BATCH_SIZE = 256
OPTIMIZER = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)

# If the dataset, embedding matrix and word count exist in the local directory
if exists(Q1_TRAINING_DATA_FILE) and exists(Q2_TRAINING_DATA_FILE) and exists(LABEL_TRAINING_DATA_FILE) and exists(NB_WORDS_DATA_FILE) and exists(WORD_EMBEDDING_MATRIX_FILE):
    # Then load them
    #q1_data = np.load(open(Q1_TRAINING_DATA_FILE, 'rb'))
    #q2_data = np.load(open(Q2_TRAINING_DATA_FILE, 'rb'))
    #labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))
    #word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
    #with open(NB_WORDS_DATA_FILE, 'r') as f:
    #    nb_words = json.load(f)['nb_words']
    pass
else:
    # Else download and extract questions pairs data
    #if not exists(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE):
    #    get_file(QUESTION_PAIRS_FILE, QUESTION_PAIRS_FILE_URL)

    #print("Processing", QUESTION_PAIRS_FILE)

    question1 = []
    question2 = []
    is_duplicate = []
    
    trn = read_snli("train.json")
    vld = read_snli('validation.json')
    tst = read_snli('test.json')    



    # Build tokenized word index
    questions = trn[0] + trn[1] + vld[0] + vld[1] + tst[0] + tst[1]
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,lower=True)
    #filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
    tokenizer.fit_on_texts(questions)
    question1_train_sequences = tokenizer.texts_to_sequences(trn[0])
    question2_train_sequences = tokenizer.texts_to_sequences(trn[1])
  
    question1_test_sequences = tokenizer.texts_to_sequences(tst[0])
    question2_test_sequences = tokenizer.texts_to_sequences(tst[1])

    question1_dev_sequences = tokenizer.texts_to_sequences(vld[0])
    question2_dev_sequences = tokenizer.texts_to_sequences(vld[1])

    word_index = tokenizer.word_index

    print("Words in index: %d" % len(word_index))

    # Download and process GloVe embeddings
    #if not exists(KERAS_DATASETS_DIR + GLOVE_ZIP_FILE):
    #    zipfile = ZipFile(get_file(GLOVE_ZIP_FILE, GLOVE_ZIP_FILE_URL))
    #    zipfile.extract(GLOVE_FILE, path=KERAS_DATASETS_DIR)

    print("Processing", GLOVE_FILE)

    embeddings_index = {}
    with open(KERAS_DATASETS_DIR + GLOVE_FILE,"r") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings: %d' % len(embeddings_index))

    # Prepare word embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
        
    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

    # Prepare training data tensors
    q1_data_train = pad_sequences(question1_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data_train = pad_sequences(question2_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    label_train = trn[2]

    q1_data_dev = pad_sequences(question1_dev_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data_dev = pad_sequences(question2_dev_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    label_dev = vld[2]

    q1_data_test = pad_sequences(question1_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data_test = pad_sequences(question2_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    label_test = tst[2]
    with open(NB_WORDS_DATA_FILE, 'w') as f:
        json.dump({'nb_words': nb_words}, f)

# Partition the dataset into train and test sets
Q1_train = q1_data_train
Q2_train = q2_data_train
Q1_test = q1_data_test
Q2_test = q2_data_test
Q1_dev = q1_data_dev
Q2_dev = q2_data_dev


def aggregate_q(input_1, num_dense=300, dropout_rate=0.5):
    feat1 = concatenate([GlobalAvgPool1D()(input_1), GlobalMaxPool1D()(input_1)])
    x = BatchNormalization()(feat1)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x



def aggregate(input_1, input_2, num_dense=300, dropout_rate=0.5):
    feat1 = concatenate([GlobalAvgPool1D()(input_1), GlobalMaxPool1D()(input_1)])
    feat2 = concatenate([GlobalAvgPool1D()(input_2), GlobalMaxPool1D()(input_2)])
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
    w_att_1 = Lambda(lambda x: softmax(x, axis=2))(attention)
    w_att_2 = Permute((2, 1))(attention)
    w_att_2 = Lambda(lambda x: softmax(x, axis=2))(w_att_2)
    in1_aligned = Dot(axes=1)([w_att_1, input_2])
    in2_aligned = Dot(axes=1)([w_att_2, input_1])
    return in2_aligned, in1_aligned

def align_q(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=2))(attention)
    in1_aligned = Dot(axes=1)([w_att_1, input_2])
    return in1_aligned



question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

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
Q = concatenate([q1,q2],axis=-2)
Encoder = Bidirectional(LSTM(units=300, return_sequences=True))
q_encoded = Dropout(DROPOUT)(Encoder(Q))

q_aligned = align_q(q_encoded, q_encoded)
q_combined = concatenate([q_encoded, q_aligned, subtract(q_encoded, q_aligned)])
compare = Bidirectional(LSTM(300, return_sequences=True))
q_compare = Dropout(DROPOUT)(compare(q_combined))

'''
q1_aligned, q2_aligned = align(q1_encoded, q2_encoded)

q1_combined = concatenate([q1_encoded, q2_aligned, subtract(q1_encoded, q2_aligned), multiply([q1_encoded, q2_aligned])])
q2_combined = concatenate([q2_encoded, q1_aligned, subtract(q2_encoded, q1_aligned), multiply([q2_encoded, q1_aligned])])
q1_combined = Dropout(DROPOUT)(q1_combined)
q2_combined = Dropout(DROPOUT)(q2_combined)
compare = Bidirectional(LSTM(300, return_sequences=True))
q1_compare = Dropout(DROPOUT)(compare(q1_combined))
q2_compare = Dropout(DROPOUT)(compare(q2_combined))
'''
x = aggregate_q(q_compare)

#x  = aggregate(q1_encoded,q2_encoded)
is_duplicate = Dense(3, activation='sigmoid')(x)
model = Model(inputs=[question1,question2], outputs=is_duplicate)
model = multi_gpu_model(model, gpus=5)
model.summary()
#model = multi_gpu_model(model, gpus=4)
model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# Train the model, checkpointing weights with best validation accuracy
print("Starting training at", datetime.datetime.now())
t0 = time.time()
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True,save_weights_only=True)]
history = model.fit([Q1_train, Q2_train],
                    label_train,
                    epochs=NB_EPOCHS,
                    validation_data=([Q1_dev,Q2_dev],label_dev),
                    verbose=2,
                    batch_size=BATCH_SIZE*5,
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
loss, accuracy = model.evaluate([Q1_test, Q2_test], label_test, verbose=0)
print('Test loss = {0:.4f}, test accuracy = {1:.4f}'.format(loss, accuracy))
