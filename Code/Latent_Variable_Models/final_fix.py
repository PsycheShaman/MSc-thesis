#-*- coding: utf-8 -*-
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

def scale(x, out_range=(0, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def load_real_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        
        with open(x_files[0],'rb') as x_file:
            x = pickle.load(x_file)
        for i in x_files[1:8]:
            if x.shape[0]<=1000000:
                with open(i,'rb') as x_file:
                    xi = pickle.load(x_file)
                    x = np.concatenate((x,xi),axis=0)
        return(x)

#with open("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\hijing-sim\\x_.pkl",'rb') as x_file:
#    geant = pickle.load(x_file)
#    
#geant.shape = (geant.shape[0],17,24,1)
gan = np.load("C:/Users/Gerhard/documents/msc-thesis-data/simulated_datasets/simulated_data_aae12/sim.npy")
        
real = load_real_data()

real.shape = (real.shape[0],17,24,1)

mx=np.max(real)
mn=np.min(real)

real = real[0:gan.shape[0],:,:,:]



real = scale(real)
gan = scale(gan)

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

epochs=5

x = np.concatenate((gan,real[0:gan.shape[0],:,:,:]),axis=0)
y = np.concatenate((np.zeros(gan.shape[0]),np.ones(real.shape[0])))
    
history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs)


import matplotlib.pyplot as plt

#plt.plot(history.history['acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/gan_vs_real__history1.png', bbox_inches='tight')
#plt.close()
#
#
#plt.plot(history.history['loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/gan_vs_real__history2.png', bbox_inches='tight')
#
#plt.close()


gan_preds = model.predict(gan)


real_preds = model.predict(real)


np.savetxt('C:/Users/gerhard/Documents/MSc-thesis-data/aae_preds.csv',gan_preds)
np.savetxt('C:/Users/gerhard/Documents/MSc-thesis-data/real_preds_v_aae.csv',real_preds)

#real = scale(real,(mn,mx))
#gan = scale(gan,(mn,mx))
#
#i = np.where(gan_preds==np.max(gan_preds))[0][0]
#p_real = gan_preds[i][0]
#plt.imshow(gan[i,:,:,0],cmap='gray')
#plt.colorbar()
#plt.title("P(real)= "+str(p_real))
#
#for(j,k) in zip((0,0.1,0.2,0.3,0.4,0.48,0.6,0.7,0.85,0.9),(0.1,0.2,0.3,0.4,0.49,0.6,0.7,0.8,0.86,1)):
#    i = np.where((gan_preds>j) & (gan_preds<=k))[0][0]
##    i = np.where(gan_preds==np.max(gan_preds))[0][0]
#    p_real = gan_preds[i][0]
#    plt.imshow(gan[i,:,:,0],cmap='gray')
#    plt.colorbar()
#    plt.title("P(real)= "+str(p_real))
#    plt.show()
#    plt.close()
#
#
#for(j,k) in zip((0.9,0.91,0.92,0.93,0.94,0.95),(0.91,0.92,0.93,0.94,0.95,0.96)):
#    m = np.where((gan_preds>j) & (gan_preds<=k))[0]
##    i = np.where(gan_preds==np.max(gan_preds))[0][0]
#    for i in m:
#        p_real = gan_preds[i]
#        plt.imshow(gan[i,:,:,0],cmap='gray')
#        plt.colorbar()
#        plt.title("P(real)= "+str(p_real[0]))
#        plt.show()
#        plt.close()
#
#
#for(j,k) in zip((0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89),(0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9)):
#    i = np.where((gan_preds>j) & (gan_preds<=k))[0][0]
##    i = np.where(gan_preds==np.max(gan_preds))[0][0]
#    p_real = gan_preds[i][0]
#    plt.imshow(gan[i,:,:,0],cmap='gray')
#    plt.colorbar()
#    plt.title("P(real)= "+str(p_real))
#    plt.show()
#    plt.close()
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#
#
#
#
#
#
