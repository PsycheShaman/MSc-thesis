# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 20:07:31 2019

@author: gerhard
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.load("C:/Users/gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")

y = np.load("C:/Users/gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

x = (x-np.max(x))/np.max(x)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, UpSampling2D, Conv2D,MaxPooling2D,LSTM,Conv1D

from tensorflow.keras.layers import Embedding, Conv3D, MaxPooling3D
from tensorflow.keras.optimizers import Adam

x_train.shape = (-1,6,17,24,1)

model = Sequential()

model.add(Conv3D(128,(3,10,10),activation='relu'))

model.add(MaxPooling3D())

model.add(Conv3D(64,(1,3,3),activation="relu"))

model.add(MaxPooling3D())

model.add(Flatten())

model.add(Dense(1024,activation='relu'))

model.add(Dense(1,activation='softmax'))

model.compile(optimizer=Adam(lr=0.0000001),loss='binary_crossentropy',metrics=['accuracy'])

batch_size=32

epochs=10

history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)













