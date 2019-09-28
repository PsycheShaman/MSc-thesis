# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:27:50 2019

@author: gerhard
"""
import numpy as np
#geant = np.load("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/geant_scaled.npy")

import glob

import numpy as np

import pickle

from ast import literal_eval

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def load_real_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        
        with open(x_files[0],'rb') as x_file:
            x = pickle.load(x_file)
        for i in x_files[1:8]:
            with open(i,'rb') as x_file:
                xi = pickle.load(x_file)
                x = np.concatenate((x,xi),axis=0)
        return(x)

with open("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\hijing-sim\\x_.pkl",'rb') as x_file:
    geant = pickle.load(x_file)
    
geant.shape = (geant.shape[0],17,24,1)
        
real = load_real_data()

real.shape = (real.shape[0],17,24,1)

real = real[0:geant.shape[0],:,:,:]

#real = scale(real)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(4,4),strides=1, padding='valid',activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(4,4),strides=1, padding='valid',activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation="sigmoid"))
model.add(tf.keras.layers.Dense(256,activation="sigmoid"))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.00001),
              metrics=['accuracy'])

batch_size=32

epochs=30

x = np.concatenate((geant,real),axis=0)
y = np.concatenate((np.zeros(geant.shape[0]),np.ones(real.shape[0])))
    
history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs)


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/geant_vs_real__history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/geant_vs_real__history2.png', bbox_inches='tight')

plt.close()


geant_preds = model.predict(geant)

plt.hist(geant_preds,bins=10,range=(0,1))
plt.title('Histogram of P(real) Predictions for Geant Data')
plt.ylabel('Frequency')
plt.xlabel('P(real)')

real_preds = model.predict(real)

plt.hist(real_preds,bins=10,range=(0,1))
plt.title('Histogram of P(real) Predictions for Real Data')
plt.ylabel('Frequency')
plt.xlabel('P(real)')

plt.imshow(np.squeeze(geant[20001]),cmap='gray')




















