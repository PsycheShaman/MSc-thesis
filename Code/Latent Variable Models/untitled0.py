# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:34:14 2019

@author: gerhard
"""

import numpy as np

sim_x = np.load("C:/Users/Gerhard/Documents/msc-thesis-data/simulated_datasets/simulated_data_aae12/sim_big.npy")

import glob, pickle

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

sim_x = sim_x[0:real.shape[0],:,:,:]

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


sim_x = scale(sim_x, out_range=(np.min(real),np.max(real)))

x = np.concatenate((sim_x,real),axis=0)
y = np.concatenate((np.zeros(sim_x.shape[0]),np.ones(real.shape[0])))

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(4,4),strides=1, padding='valid',activation="relu",input_shape=(17,24,1)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(4,4),strides=1, padding='valid',activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation="sigmoid"))
model.add(tf.keras.layers.Dense(256,activation="sigmoid"))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model,to_file="C:/Users/gerhard.22seven/Desktop/sim_disx.png",show_shapes=True,show_layer_names=False)



model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.00001),
              metrics=['accuracy'])

batch_size=32

epochs=10



history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs)


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/vae_vs_real__FINAL1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/vae_vs_real__FINAL2.png', bbox_inches='tight')

plt.close()

sim_preds = model.predict(sim_x)
sim_y = np.zeros(len(sim_preds))

preds = np.c_[sim_preds,sim_y]

real_preds = model.predict(real)
real_y = np.ones(len(real_preds))

preds = np.r_[preds,np.c_[real_preds,real_y]]

sim_preds.shape = len(sim_preds)
real_preds.shape = len(real_preds)

colors = ['red','blue']
plt.hist([sim_preds,real_preds],bins=100,color=colors,label=['AAE','real'],histtype='bar',stacked=True)
plt.legend(loc=0)
plt.xlabel("P(real)")
plt.ylabel("Frequency")

ind = np.where((sim_preds>0.9) & (sim_preds<=0.95))[0][0]
ind = np.where(sim_preds==np.max(sim_preds))[0][0]

a = sim_x[ind,:,:,0]
p = sim_preds[ind]

plt.imshow(a,cmap='gray')
plt.axis('off')
plt.colorbar()
plt.title('AAE: P(real) = %f' % p)


############################################################################3

import numpy as np

sim_x = np.load("C:/Users/Gerhard/Documents/msc-thesis-data/simulated_datasets/simulated_data_gan_16/1million_after200000epochs.npy")


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

sim_x = sim_x[0:real.shape[0],:,:,:]

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


sim_x = scale(sim_x, out_range=(np.min(real),np.max(real)))

x = np.concatenate((sim_x,real),axis=0)
y = np.concatenate((np.zeros(sim_x.shape[0]),np.ones(real.shape[0])))

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

epochs=10



history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs)


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/gan_vs_real__FINAL1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/gan_vs_real__FINAL2.png', bbox_inches='tight')

plt.close()

sim_preds = model.predict(sim_x)
sim_y = np.zeros(len(sim_preds))

preds = np.c_[sim_preds,sim_y]

real_preds = model.predict(real)
real_y = np.ones(len(real_preds))

preds = np.r_[preds,np.c_[real_preds,real_y]]

sim_preds.shape = len(sim_preds)
real_preds.shape = len(real_preds)

colors = ['red','blue']
plt.hist([sim_preds,real_preds],bins=100,color=colors,label=['GAN','real'],histtype='bar',stacked=True)
plt.legend(loc=0)
plt.xlabel("P(real)")
plt.ylabel("Frequency")

ind = np.where((sim_preds>0.00002) & (sim_preds<0.00003))[0][0]
#ind = np.where(sim_preds==np.max(sim_preds))[0][0]

a = sim_x[ind,:,:,0]
p = sim_preds[ind]

plt.imshow(a,cmap='gray')
plt.axis('off')
plt.colorbar()
plt.title('GAN: P(real) = %f' % p)

############################################################################################
import numpy as np

#sim_x = np.load("C:/Users/Gerhard/Documents/msc-thesis-data/generated-data/vae11/0.npy")

sim_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data/generated-data/vae11/*.npy")
with open(sim_files[0],'rb') as x_file:
    sim_x = np.load(x_file)
for i in sim_files[1:]:
    with open(i,'rb') as x_file:
        xi = np.load(x_file)
        sim_x = np.concatenate((sim_x,xi),axis=0)


def load_real_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        
        with open(x_files[0],'rb') as x_file:
            x = pickle.load(x_file)
        for i in x_files[1:3]:
            with open(i,'rb') as x_file:
                xi = pickle.load(x_file)
                x = np.concatenate((x,xi),axis=0)
        return(x)
        
real = load_real_data()

real.shape = (real.shape[0],17,24,1)
sim_x.shape = (sim_x.shape[0],17,24,1)

#sim_x = sim_x[0:real.shape[0],:,:,:]

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


sim_x = scale(sim_x, out_range=(np.min(real),np.max(real)))

x = np.concatenate((sim_x,real),axis=0)
y = np.concatenate((np.zeros(sim_x.shape[0]),np.ones(real.shape[0])))

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

epochs=10



history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs)


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/vae_vs_real__FINAL1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/vae_vs_real__FINAL2.png', bbox_inches='tight')

plt.close()

sim_preds = model.predict(sim_x)
sim_y = np.zeros(len(sim_preds))

preds = np.c_[sim_preds,sim_y]

real_preds = model.predict(real)
real_y = np.ones(len(real_preds))

preds = np.r_[preds,np.c_[real_preds,real_y]]

sim_preds.shape = len(sim_preds)
real_preds.shape = len(real_preds)

colors = ['red','blue']
plt.hist([sim_preds,real_preds],bins=100,color=colors,label=['VAE','real'],histtype='bar',stacked=True)
plt.legend(loc=0)
plt.xlabel("P(real)")
plt.ylabel("Frequency")

ind = np.where((sim_preds==0))[0][1]
#ind = np.where(sim_preds==np.max(sim_preds))[0][0]

a = sim_x[ind,:,:,0]
p = sim_preds[ind]

plt.imshow(a,cmap='gray')
plt.axis('off')
plt.colorbar()
plt.title('VAE: P(real) = %2.e' % p)











