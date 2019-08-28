# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:08:53 2019

@author: gerhard
"""

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("C:/Users/gerhard/documents/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("C:/Users/gerhard/documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24, 1))


labels = np.repeat(infosets[:, 0], 6)

#train = (train-np.max(train))/np.max(train)

#from sklearn.model_selection import train_test_split
#
#x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.33, random_state=42)
#
#x_train = x_train.reshape((-1, 17, 24, 1))
#
#x_test = x_test.reshape((-1, 17, 24, 1))

import tensorflow as tf

x2 = (train-np.max(train))/np.max(train)

x = np.fft.fft2(train)

real = np.real(x)

imaginary = np.imag(x)

#x = np.concatenate((real,imaginary,x),axis=3)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=1, padding='same',activation="relu",name="16filters8x2",input_shape=(17,24,1)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=1, padding='same',activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=1, padding='same',activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=1, padding='same',activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(256,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(64,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

#from keras.utils import plot_model
#plot_model(model, to_file='C:/Users/gerhard/Documents/MSc-thesis/model4.png',show_shapes=True,show_layer_names=False)
#
batch_size=512

epochs=10

history=model.fit(real, labels,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=2,
              validation_split=.1)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/documents/fft1_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/documents/fft1_history2.png', bbox_inches='tight')

plt.close()

















