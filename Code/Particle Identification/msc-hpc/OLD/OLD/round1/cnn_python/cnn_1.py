# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:01:37 2019

@author: Gerhard
"""

print("==============================================================================================")

import pickle

import glob

import numpy as np

x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/x_*.pkl")
y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/y_*.pkl")

#x_files = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/cnn/x_*.pkl")
#y_files = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/cnn/y_*.pkl")

print("loading first x pickle........................................................................................")

with open(x_files[0], 'rb') as x_file0:
    x = pickle.load(x_file0)
    
print("loading first y pickle........................................................................................")

with open(y_files[0], 'rb') as y_file0:
   y = pickle.load(y_file0)
   
print("recursively adding x pickles........................................................................................")

for i in x_files[1:]:
#for i in x_files[1:2]:
    with open(i,'rb') as x_file:
        xi = pickle.load(x_file)
        x = np.concatenate((x,xi),axis=0)
        
print("recursively adding y pickles........................................................................................")
        
for i in y_files[1:]:
#for i in y_files[1:2]:
    with open(i,'rb') as y_file:
        yi = pickle.load(y_file)
        y = np.concatenate((y,yi),axis=None)

#remove 0 elements
    
nz = np.array([np.count_nonzero(i) for i in x])

zeros = np.where(nz==0)

x = np.delete(x,zeros,axis=0)
y = np.delete(y,zeros)

#oversample electrons

elec = np.where(y==1)
pion = np.where(y!=1)

int(elec[0].shape[0])/int(pion[0].shape[0])

electrons_x = x[elec,:,:]

electrons_y = y[elec]

electrons_x = np.squeeze(electrons_x)

x = np.concatenate((electrons_x,x,electrons_x),axis=0)

y = np.concatenate((electrons_y,y,electrons_y),axis=None)

x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)

mu = np.mean(x)

x = np.true_divide(x,mu)
 
#with open('/scratch/vljchr004/data/msc-thesis-data/cnn/x.pkl', 'rb') as x_file:
#    x = pickle.load(x_file)
#
#with open('/scratch/vljchr004/data/msc-thesis-data/cnn/y.pkl', 'rb') as y_file:
#    y = pickle.load(y_file)

#with open('C:/Users/gerhard/Documents/msc-thesis-data/cnn/x_000265309.pkl', 'rb') as x_file:
#    x = pickle.load(x_file)
#
#with open('C:/Users/gerhard/Documents/msc-thesis-data/cnn/y_000265309.pkl', 'rb') as y_file:
#    y = pickle.load(y_file)
    
#x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=123456)

import tensorflow

from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

num_classes = 2
epochs = 100

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(17,24,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
              #batch_size=batch_size,
              epochs=epochs,
              validation_split=0.15,
              shuffle=True,
              verbose=2)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/msc-hpc/cnn_python/fig/cnn_1_history1.png', bbox_inches='tight')

#plt.savefig('C:/Users/gerhard/Documents/msc-hpc/cnn_python/fig/test/cnn_1_history1.png', bbox_inches='tight')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/msc-hpc/cnn_python/fig/cnn_1_history2.png', bbox_inches='tight')

#plt.savefig('C:/Users/gerhard/Documents/msc-hpc/cnn_python/fig/test/cnn_1_history2.png', bbox_inches='tight')

model.probs = model.predict_proba(x_test)

np.savetxt("/home/vljchr004/msc-hpc/cnn_python/results/cnn_1_results.csv", np.array(model.probs), fmt="%s")

#np.savetxt("C:/Users/gerhard/Documents/msc-hpc/cnn_python/results/test/cnn_1_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("/home/vljchr004/msc-hpc/cnn_python/results/cnn_1_y_test.csv", np.array(y_test), fmt="%s")

#np.savetxt("C:/Users/gerhard/Documents/msc-hpc/cnn_python/results/test/cnn_1_y_test.csv", np.array(y_test), fmt="%s")

model.save('/home/vljchr004/msc-hpc/cnn_python/cnn1.h5')  # creates a HDF5 file 'my_model.h5'

#model.save('C:/Users/gerhard/Documents/msc-hpc/cnn_python/results/test/cnn1.h5')

del model

print("<----*********************+++++++++done+++++++++++*******************---->")


















