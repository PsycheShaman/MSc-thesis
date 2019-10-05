print("==============================================================================================")

print("starting........................................................................................")

import glob

import numpy as np

print("imported glob, np........................................................................................")

x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/x_*.pkl")
y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/y_*.pkl")

#x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
#y_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\y_*.pkl")

import pickle

print("loading first x pickle........................................................................................")

with open(x_files[0], 'rb') as x_file0:
    x = pickle.load(x_file0)
    
print("loading first y pickle........................................................................................")

with open(y_files[0], 'rb') as y_file0:
   y = pickle.load(y_file0)
   
print("recursively adding x pickles........................................................................................")

for i in x_files[1:]:
    with open(i,'rb') as x_file:
        print(i)
        xi = pickle.load(x_file)
        x = np.concatenate((x,xi),axis=0)
        
print("recursively adding y pickles........................................................................................")
        
for i in y_files[1:]:
    with open(i,'rb') as y_file:
        yi = pickle.load(y_file)
        y = np.concatenate((y,yi),axis=None)
 
x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/x_*.npy")
y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/y_*.npy")
       
print("recursively adding x numpys........................................................................................")

for i in x_files[0:]:
    with open(i,'rb') as x_file:
        print(i)
        xi = np.load(x_file)
        x = np.concatenate((x,xi),axis=0)

print("recursively adding y numpys........................................................................................")

for i in y_files[0:]:
    with open(i,'rb') as y_file:
        yi = np.load(y_file)
        y = np.concatenate((y,yi),axis=None)        

print("removing useless elements........................................................................................")

nz = np.array([np.count_nonzero(i) for i in x])

zeros = np.where(nz==0)

x = np.delete(x,zeros,axis=0)
y = np.delete(y,zeros)

#oversample electrons

#elec = np.where(y==1)
#pion = np.where(y!=1)
#
#electrons_x = x[elec,:,:]
#
#electrons_y = y[elec]
#
#electrons_x = np.squeeze(electrons_x)
#
#pion = pion[0:electrons_x.shape[0]]
#
#pions_x = x[pion,:,:]
#
#pions_y = y[pion]
#
#pions_x = np.squeeze(pions_x)
#
#x = np.concatenate((electrons_x,pions_x),axis=0)
#
#y = np.concatenate((electrons_y,pions_y),axis=None)

x.shape = (x.shape[0],x.shape[1],x.shape[2],1)

#from sklearn.preprocessing import StandardScaler
#
#scaler = StandardScaler()
#
#scaler.fit(x.astype(float))
#
#x = scaler.transform(x.astype(float))

mu = np.mean(x)

sd = np.std(x)

x = x-mu

x=x/sd


#from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=123456)

#class_weights = class_weight.compute_class_weight('balanced',
#                                                 np.unique(y_train),
#                                                 y_train)
#
#class_weights = {0:class_weights[0],1:class_weights[1]}

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



import tensorflow

from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


#num_classes = 2
epochs = 20

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='tanh',
                 input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]),data_format="channels_last"))
model.add(Conv2D(64, (3, 3), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

sgd = tensorflow.keras.optimizers.SGD(lr=0.01) 

# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

batch_size=10000
    
history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
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
plt.savefig('/home/vljchr004/msc-hpc/round3/conv/round3_model5_history1.png', bbox_inches='tight')
# summarize history for loss

plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/msc-hpc/round3/conv/round3_model5_history2.png', bbox_inches='tight')

model.probs = model.predict_proba(x_test)

import numpy as np
np.savetxt("/home/vljchr004/msc-hpc/round3/conv/round3_model5_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("/home/vljchr004/msc-hpc/round3/conv/round3_model5_y_test.csv", np.array(y_test), fmt="%s")

model.save('/home/vljchr004/msc-hpc/round3/conv/round3_model5_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done------------------------------------------>")

