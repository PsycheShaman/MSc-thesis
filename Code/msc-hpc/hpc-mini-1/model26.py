# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 18:47:05 2019

@author: gerhard
"""

import glob

import numpy as np

#P_files = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/P_*.pkl", recursive=True)

x_files = glob.glob("/scratch/vljchr004/1_8_to_2_2_GeV/x_*.pkl")
y_files = glob.glob("/scratch/vljchr004/1_8_to_2_2_GeV/y_*.pkl")

#x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
#y_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\y_*.pkl")

import pickle

print("loading first x pickle........................................................................................")

with open(x_files[0], 'rb') as x_file0:
    x = pickle.load(x_file0)
    
print("loading first y pickle........................................................................................")

with open(y_files[0], 'rb') as y_file0:
   y = pickle.load(y_file0)

#with open(P_files[0], 'rb') as P_file0:
#   P = pickle.load(P_file0)

x.shape = (x.shape[1],x.shape[2],x.shape[3])
print("x.shape")
print(x.shape)
   
print("recursively adding x pickles........................................................................................")

for i in x_files[1:]:
    with open(i,'rb') as x_file:
        print(i)
        xi = pickle.load(x_file)
        xi.shape = (xi.shape[1],xi.shape[2],xi.shape[3])
        print("xi.shape")
        print(xi.shape)
        x = np.concatenate((x,xi),axis=0)
        
print("recursively adding y pickles........................................................................................")
        
for i in y_files[1:]:
    with open(i,'rb') as y_file:
        yi = pickle.load(y_file)
        y = np.concatenate((y,yi),axis=None)
        
#for i in P_files[1:]:
#    with open(i,'rb') as P_file:
#        Pi = pickle.load(P_file)
#        P = np.concatenate((P,Pi),axis=None)
        
#x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/x_*.npy")
#y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/y_*.npy")
#       
#print("recursively adding x numpys........................................................................................")
#
#for i in x_files[0:]:
#    with open(i,'rb') as x_file:
#        print(i)
#        xi = np.load(x_file)
#        x = np.concatenate((x,xi),axis=0)
#
#print("recursively adding y numpys........................................................................................")
#
#for i in y_files[0:]:
#    with open(i,'rb') as y_file:
#        yi = np.load(y_file)
#        y = np.concatenate((y,yi),axis=None)
        
nz = np.array([np.count_nonzero(i) for i in x])

zeros = np.where(nz==0)

x = np.delete(x,zeros,axis=0)
y = np.delete(y,zeros)
#P = np.delete(P,zeros)

#x.shape = (x.shape[0],x.shape[1],x.shape[2],1)

x.shape = (x.shape[0],x.shape[2],x.shape[1])

print("x.shape after reshape for lstm")
print(x.shape)

#GeV_range2 = np.where(P>=1.8 and P<=2.2)
#
#x = x[GeV_range2,:,:,:]
#y = y[GeV_range2]


electrons = np.where(y==1)

electrons = electrons[0]

pions = np.where(y==0)

pions = pions[0]

pions = pions[0:electrons.shape[0]]

x_1 = x[electrons,:,:]
x_2 = x[pions,:,:]

x = np.vstack((x_1,x_2))

y_1 = y[electrons]
y_2 = y[pions]

y = np.concatenate((y_1,y_2),axis=None)

#ma = np.max(x)
#
#x = x/ma

x=x.astype('float')

for i in range(0,x.shape[0]):
    ma = np.max(x[i,:,:])
    x[i,:,:]=x[i,:,:]/ma

#ma = np.amax(x,axis=2)
#
#x = np.divide(x,ma)

#check the division above before running!!!!!!!!!!!1

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=123456)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



import tensorflow

from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Bidirectional, TimeDistributed

model = Sequential()
model.add(LSTM(256,input_shape=(24,17),return_sequences=True))
model.add(LSTM(256,go_backwards=True,return_sequences=False))
model.add(Dense(256,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dense(2,activation="softmax"))

sgd = tensorflow.keras.optimizers.SGD(momentum=0.9) 

# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

batch_size=32

epochs=50
    
history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)#,
              #class_weight=class_weights)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/hpc-mini/model26_history1.png', bbox_inches='tight')
# summarize history for loss

plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/hpc-mini/model26_history2.png', bbox_inches='tight')

model.probs = model.predict_proba(x_test)

import numpy as np
np.savetxt("/home/vljchr004/hpc-mini/model26_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("/home/vljchr004/hpc-mini/model26_y_test.csv", np.array(y_test), fmt="%s")

model.save('/home/vljchr004/hpc-mini/model26_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done------------------------------------------>")











