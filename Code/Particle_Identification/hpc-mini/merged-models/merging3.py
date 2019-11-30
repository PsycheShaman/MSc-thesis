# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:09:07 2019

@author: gerhard
"""

import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

def load_data():
        tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

        infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

        x = tracks.reshape((-1, 17,24,1))

#        x = tracks

        y = np.repeat(infosets[:, 0], 6)
        return (x,y)

(x,y) = load_data()

x = (x-np.max(x))/np.max(x)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, UpSampling2D, Conv2D,MaxPooling2D,LSTM,Conv1D

from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam

LSTM_X_train,LSTM_X_test,LSTM_y_train,LSTM_y_test = X_train, X_test, y_train, y_test

LSTM_X_train = LSTM_X_train.reshape((-1,24,17))
LSTM_X_test = LSTM_X_test.reshape((-1,24,17))

collapse_X_train,collapse_X_test,collapse_y_train,collapse_y_test = X_train, X_test, y_train, y_test

flatten_X_train,flatten_X_test,flatten_y_train,flatten_y_test = X_train, X_test, y_train, y_test

flatten_X_train = flatten_X_train.reshape((-1,17*24))
flatten_X_test = flatten_X_test.reshape((-1,17*24))


def collapse(x):
    newx  = []

    for i in range(0,x.shape[0]):
        xi = x[i,:,:].sum(axis=0)
        newx.append(xi)

    newx = np.array(newx)

    newx.shape = (newx.shape[0],newx.shape[1],1)
    return newx

collapse_X_train = collapse(collapse_X_train)
collapse_X_test = collapse(collapse_X_test)


a = Input(shape=(17,24,1))
b = Input(shape=(24,17))
c = Input(shape=(24,1))
d = Input(shape=(17*24,))

#Convolutions on dataset a

#conv = Conv2D(32,(3,3),padding='same')
#pool = MaxPooling2D()
#flatten = Flatten()

conved_a = Conv2D(32,(3,3),padding='same',activation='sigmoid')(a)
conved_a = MaxPooling2D()(conved_a)
conved_a = Conv2D(64,(3,3),padding='same',activation='sigmoid')(conved_a)
conved_a = MaxPooling2D()(conved_a)
flattened_a = Flatten()(conved_a)
dense_a =  Dense(256,activation=tf.nn.leaky_relu)(flattened_a)
dense_a =  Dense(128,activation=tf.nn.leaky_relu)(dense_a)
dense_a =  Dense(16,activation=tf.nn.leaky_relu)(dense_a)


#LSTM on dataset b

lstm_b = LSTM(256,return_sequences=True)(b)
lstm_b = LSTM(256)(lstm_b)

dense_b = Dense(256,activation=tf.nn.leaky_relu)(lstm_b)
dense_b = Dense(128,activation=tf.nn.leaky_relu)(dense_b)
dense_b = Dense(16,activation=tf.nn.leaky_relu)(dense_b)

#1Dconv on dataset c

conv1 = Conv1D(32,3)

conved_c = conv1(c)

flattened_c = Flatten()(conved_c)

dense_c =  Dense(256,activation=tf.nn.leaky_relu)(flattened_c)

dense_c =  Dense(128,activation=tf.nn.leaky_relu)(dense_c)

dense_c =  Dense(16,activation=tf.nn.leaky_relu)(dense_c)

dense_d = Dense(512,activation=tf.nn.leaky_relu)(d)

dense_d = Dense(256,activation=tf.nn.leaky_relu)(dense_d)

dense_d = Dense(128,activation=tf.nn.leaky_relu)(dense_d)

dense_d = Dense(16,activation=tf.nn.leaky_relu)(dense_d)

merge = tf.keras.layers.concatenate([dense_a,dense_b,dense_c,dense_d])

dense1 = Dense(512,activation=tf.nn.leaky_relu)(merge)

dense2 = Dense(256,activation=tf.nn.leaky_relu)(dense1)

dense2 = Dense(128,activation=tf.nn.leaky_relu)(dense2)

dense2 = Dense(16,activation=tf.nn.leaky_relu)(dense2)

out = Dense(1,activation="sigmoid")(dense2)


model = Model(inputs=[a,b,c,d],outputs=out)

model.compile(optimizer=Adam(lr=0.0000001),loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit([X_train,LSTM_X_train,collapse_X_train,flatten_X_train],y_train,epochs=100,batch_size=32,validation_split=0.25,shuffle=True)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/hpc-mini/merged-models/model2_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/hpc-mini/chamber_gain_corrected/model2_history2.png', bbox_inches='tight')

plt.close()


model.probs = model.predict([X_test,LSTM_X_test,collapse_X_test,flatten_X_test])


np.savetxt("/home/vljchr004/hpc-mini/merged-models/merging2_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("/home/vljchr004/hpc-mini/merged-models/merging2_y_test.csv", np.array(y_test), fmt="%s")
