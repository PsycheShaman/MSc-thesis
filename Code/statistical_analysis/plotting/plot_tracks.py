# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 22:34:03 2019

@author: gerhard
"""

import numpy as np

import glob

real_x = np.load("C:/Users/gerhard/Documents/msc-thesis-data/cnn/x_000265343.npy")

real_y = np.load("C:/Users/gerhard/Documents/msc-thesis-data/cnn/y_000265343.npy")

fake_files = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/hijing-sim/*tracks.npy")

sim_x = np.load(fake_files[0])

for i in fake_files[1:]:
    this = np.load(i)
    sim_x = np.concatenate((sim_x,this),axis=0)

sim_x.shape = (sim_x.shape[0]*sim_x.shape[1],17,24)

nz = np.array([np.count_nonzero(i) for i in real_x])

zeros = np.where(nz==0)

real_x = np.delete(real_x,zeros,axis=0)
real_y = np.delete(real_y,zeros)

nz = np.array([np.count_nonzero(i) for i in sim_x])

zeros = np.where(nz==0)

sim_x = np.delete(sim_x,zeros,axis=0)

sim_y = np.zeros(sim_x.shape[0])

real_pions = np.where(real_y==1)

real_x = real_x[real_pions[0],:,:]

real_y = np.ones(real_x.shape[0])

real_x = real_x[0:sim_x.shape[0],:,:]
real_y = real_y[0:sim_x.shape[0]]

make_zero = np.where(sim_x==-7169)

sim_x[make_zero] = 0

x = np.concatenate((real_x,sim_x))

y = np.concatenate((real_y,sim_y))

x.shape = (-1,17,24,1)

import tensorflow as tf

import matplotlib.pyplot as plt

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=1, padding='valid',activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=1, padding='valid',activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(256,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=30

history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/geant_vs_real/model0_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/geant_vs_real/model0_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/geant_vs_real/model0_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/geant_vs_real/model0_y_test.csv", np.array(y), fmt="%s")

model.save('C:/Users/gerhard/Documents/hpc-mini/geant_vs_real/model0_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done------------------------------------------>")










