# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:39:15 2019

@author: gerhard
"""

import glob

import numpy as np

import pickle

from ast import literal_eval

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

geant = np.load("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/geant_scaled.npy")
aae = np.load("c:/Users/gerhard/Documents//msc-thesis-data/simulated_datasets/simulated_data_aae12/sim.npy")
gan = np.load("c:/Users/gerhard/Documents//msc-thesis-data/simulated_datasets/simulated_data_gan_16/1million_after200000epochs.npy")
vae = np.load("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/vae_scaled.npy")



def load_real_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        
        with open(x_files[0],'rb') as x_file:
            x = pickle.load(x_file)
        for i in x_files[1:8]:
            with open(i,'rb') as x_file:
                xi = pickle.load(x_file)
                x = np.concatenate((x,xi),axis=0)
        return(x)
        
real = load_real_data()



real.shape = (real.shape[0],17,24,1)

gan = scale(gan)
aae = scale(aae)
vae = scale(vae)
geant = scale(geant)
real = scale(real)

aae_train = aae[0:75000,:,:,:]
aae_test = aae[75001:100000,:,:,:]

gan_train = gan[0:75000,:,:,:]
gan_test = gan[75001:100000,:,:,:]

vae_train = vae[0:75000,:,:,:]
vae_test = vae[75001:100000,:,:,:]

geant_train = geant[0:75000,:,:,:]
geant_test = geant[75001:100000,:,:,:]


sim_train = np.concatenate((aae_train,vae_train,geant_train,gan_train))

real_train = real[0:sim_train.shape[0],:,:,:]
real_test = real[sim_train.shape[0]:,:,:,:]


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

epochs=20

x = np.concatenate((sim_train,real_train),axis=0)
y = np.concatenate((np.zeros(sim_train.shape[0]),np.ones(real_train.shape[0])))
    
history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs)


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/aae_vs_real__history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/aae_vs_real__history2.png', bbox_inches='tight')

plt.close()

geant_pred = model.predict_proba(geant_test)
aae_pred = model.predict_proba(aae_test)
vae_pred = model.predict_proba(vae_test)
real_pred = model.predict_proba(real_test)
gan_pred = model.predict_proba(gan_test)

np.save("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/geant_preds.npy",geant_pred)
np.save("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/aae_preds.npy",aae_pred)
np.save("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/vae_preds.npy",vae_pred)
np.save("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/real_preds.npy",real_pred)
np.save("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/gan_preds.npy",gan_pred)


np.mean(geant_pred)
np.mean(aae_pred)
np.mean(vae_pred)
np.mean(gan_pred)
np.mean(real_pred)

data_to_plot = [geant_pred, aae_pred, vae_pred, gan_pred, real_pred]

plt.boxplot(data_to_plot)


plt.xticks([1, 2, 3, 4, 5], ['Geant', 'AAE', 'VAE', 'GAN', 'Real'])
plt.ylabel("P(real)")
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/generative_models_compared.png', bbox_inches='tight')
plt.close()









