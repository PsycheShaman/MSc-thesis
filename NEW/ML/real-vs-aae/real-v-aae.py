# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:39:15 2019

@author: gerhard
"""

import glob

import numpy as np

import pickle

def load_real_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        
        with open(x_files[0],'rb') as x_file:
            x = pickle.load(x_file)
        return(x)


def load_simulated_data():
    x_files = glob.glob("C:\\Users\\gerhard\\Documents\\Keras-GAN\\newAAE\\v11\\simulated_data\\*.npy")
    
    with open(x_files[0],'rb') as x_file:
        x = np.load(x_file)
        
#len(x_files)-10000:len(x_files)
        
    for i in x_files[1:]:
        with open(i,'rb') as x_file:
            xi = np.load(x_file)
            x = np.concatenate((x,xi),axis=0)
    return(x)

sim = load_simulated_data()
real = load_real_data()

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

real = real[0:sim.shape[0],:,:]
real.shape = (sim.shape[0],17,24,1)

sim = scale(sim)
real = scale(real)

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

epochs=100

x = np.concatenate((sim,real),axis=0)
y = np.concatenate((np.zeros(sim.shape[0]),np.ones(sim.shape[0])))
    
history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/aae_vs_real__history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/aae_vs_real__history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)























