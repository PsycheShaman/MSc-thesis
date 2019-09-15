# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:39:15 2019

@author: gerhard
"""

import glob

import numpy as np

import pickle

from ast import literal_eval

#def load_real_data():
#        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
#        
#        with open(x_files[0],'rb') as x_file:
#            x = pickle.load(x_file)
#        for i in x_files[1:5]:
#            with open(i,'rb') as x_file:
#                xi = pickle.load(x_file)
#                x = np.concatenate((x,xi),axis=0)
#        return(x)
#        
#real = load_real_data()
#
#real.shape = (real.shape[0],17,24,1)
#
#
#def load_aae_data():
#    x_files = glob.glob("C:\\Users\\gerhard\\Documents\\Keras-GAN\\newAAE\\v11\\simulated_data\\*.npy")
#    
#    with open(x_files[len(x_files)-10000],'rb') as x_file:
#        x = np.load(x_file)
#        
#    for i in x_files[len(x_files)-10000+1:]:
#        with open(i,'rb') as x_file:
#            xi = np.load(x_file)
#            x = np.concatenate((x,xi),axis=0)
#    return(x)
#
#def load_vae_data():
#    x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\generated-data\\vae11\\*.npy")
#    
#    with open(x_files[0],'rb') as x_file:
#        x = np.load(x_file)
#        
#    for i in x_files[1:]:
#        with open(i,'rb') as x_file:
#            xi = np.load(x_file)
#            x = np.concatenate((x,xi),axis=0)
#    return(x)
#    
#
#
def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
#
#
#
#
#
#
#def load_geant_data():
#    x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\hijing-sim\\test3\\test2\\**\\*.txt"\
#                        , recursive=True)
#    def file_reader2(i,l):
#        di = open(i)
#        di = di.read()
#        if di == "}":
#            pass
#        else:
##            di = di + "}"
#            di = literal_eval(di)
#            ki = list(di.keys())
#            layer = [di.get(k).get(l) for k in ki]
##            print(i)
#            return(layer)
#            
#    layer0 = [file_reader2(i,"layer 0") for i in x_files]
#    layer0 = np.array([item for sublist in layer0 for item in sublist])
#    empties = np.where([np.array(i).shape!=(17,24) for i in layer0])
#    layer0 = np.delete(layer0, empties)
#    layer0 = np.stack(layer0)
#    
#    layer1 = [file_reader2(i,"layer 1") for i in x_files]
#    layer1 = np.array([item for sublist in layer1 for item in sublist])
#    empties = np.where([np.array(i).shape!=(17,24) for i in layer1])
#    layer1 = np.delete(layer1, empties)
#    layer1 = np.stack(layer1)
#    
#    layer2 = [file_reader2(i,"layer 2") for i in x_files]
#    layer2 = np.array([item for sublist in layer2 for item in sublist])
#    empties = np.where([np.array(i).shape!=(17,24) for i in layer2])
#    layer2 = np.delete(layer2, empties)
#    layer2 = np.stack(layer2)
#    
#    layer3 = [file_reader2(i,"layer 3") for i in x_files]
#    layer3 = np.array([item for sublist in layer3 for item in sublist])
#    empties = np.where([np.array(i).shape!=(17,24) for i in layer3])
#    layer3 = np.delete(layer3, empties)
#    layer3 = np.stack(layer3)
#    
#    layer4 = [file_reader2(i,"layer 4") for i in x_files]
#    layer4 = np.array([item for sublist in layer4 for item in sublist])
#    empties = np.where([np.array(i).shape!=(17,24) for i in layer4])
#    layer4 = np.delete(layer4, empties)
#    layer4 = np.stack(layer4)
#    
#    layer5 = [file_reader2(i,"layer 5") for i in x_files]
#    layer5 = np.array([item for sublist in layer5 for item in sublist])
#    empties = np.where([np.array(i).shape!=(17,24) for i in layer5])
#    layer5 = np.delete(layer5, empties)
#    layer5 = np.stack(layer5)
#    
#    x = np.concatenate((layer0,layer1,layer2,layer3,layer4,layer5),axis=0)
#    return(x)
    
#aae = load_aae_data()
#print("loaded AAE")
#vae = load_vae_data()
#print("loaded VAE")
#real = load_real_data()
#print("loaded REAL")
#
#aae = scale(aae)
#vae = scale(vae)
#real = scale(real)
#
#aae.shape = (aae.shape[0],17,24,1)
#vae.shape = (vae.shape[0],17,24,1)
#real.shape = (real.shape[0],17,24,1)
#    
#geant = load_geant_data()
#
#print("loaded GEANT")
#
#geant.shape = (geant.shape[0],17,24,1)
#
#geant = scale(geant)

#np.save("c:/Users/gerhard/Documents/MSc-thesis/simulated_datasets/geant_scaled.npy",geant)
#np.save("c:/Users/gerhard/Documents/MSc-thesis/simulated_datasets/aae_scaled.npy",aae)
#np.save("c:/Users/gerhard/Documents/MSc-thesis/simulated_datasets/vae_scaled.npy",vae)
#np.save("c:/Users/gerhard/Documents/MSc-thesis/simulated_datasets/real_scaled.npy",real)

geant = np.load("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/geant_scaled.npy")
aae = np.load("c:/Users/gerhard/Documents/Keras-GAN/newAAE/v12/simulated_data/sim.npy")
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

#real = np.load("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/real_scaled.npy")

aae = aae[0:175303,:,:,:]

#real = scale(real)
geant = scale(geant)
vae = scale(vae)
aae = scale(aae)


sim = np.concatenate((aae,vae,geant))

real.shape = (real.shape[0],17,24,1)

if real.shape[0]> sim.shape[0]:
    real = real[0:sim.shape[0],:,:,:]
    
real = scale(real)
#geant = scale(geant)
#vae = scale(vae)
#aae = scale(aae)


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

epochs=50

x = np.concatenate((sim,real),axis=0)
y = np.concatenate((np.zeros(sim.shape[0]),np.ones(real.shape[0])))
    
history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

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

geant_pred = model.predict_proba(geant)
aae_pred = model.predict_proba(aae)
vae_pred = model.predict_proba(vae)
real_pred = model.predict_proba(real)

np.save("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/geant_preds.npy",geant_pred)
np.save("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/aae_preds.npy",aae_pred)
np.save("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/vae_preds.npy",vae_pred)
np.save("c:/Users/gerhard/Documents/msc-thesis-data/simulated_datasets/real_preds.npy",real_pred)


np.mean(geant_pred)
np.mean(aae_pred)
np.mean(vae_pred)
np.mean(real_pred)

data_to_plot = [geant_pred, aae_pred, vae_pred, real_pred]


# Create a figure instance
#fig = plt.figure(1, figsize=(9, 6))

#plt.set_xticklabels(['Geant', 'AAE', 'VAE', 'Real'])

plt.boxplot(data_to_plot)


plt.xticks([1, 2, 3,4], ['Geant', 'AAE', 'VAE', 'Real'])
plt.ylabel("P(real)")










