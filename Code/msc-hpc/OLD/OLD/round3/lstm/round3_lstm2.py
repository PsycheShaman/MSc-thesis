print("==============================================================================================")

print("starting........................................................................................")

import glob

import numpy as np

print("imported glob, np........................................................................................")

x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/x_*.pkl")
y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/y_*.pkl")

#x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\ff\\x_*.pkl")
#y_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\ff\\y_*.pkl")

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
        xi = pickle.load(x_file)
        x = np.concatenate((x,xi),axis=0)
        
print("recursively adding y pickles........................................................................................")
        
for i in y_files[1:]:
    with open(i,'rb') as y_file:
        yi = pickle.load(y_file)
        y = np.concatenate((y,yi),axis=None)
        
nz = np.array([np.count_nonzero(i) for i in x])

zeros = np.where(nz==0)

x = np.delete(x,zeros,axis=0)
y = np.delete(y,zeros)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x.astype(float))

x = scaler.transform(x.astype(float))

x.shape = (x.shape[0],x.shape[1],1)


from sklearn.utils import class_weight

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
from tensorflow.keras.layers import LSTM, MaxPooling1D, Conv1D

epochs = 20


from tensorflow.keras import optimizers

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)   

model = Sequential()
model.add(Conv1D(filters=32,kernel_size=4,input_shape=(24,1)))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.5))
#model.add(Flatten())
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32,return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='softmax'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])



history = model.fit(x_train, y_train,
              batch_size=1000,
              epochs=epochs,
              validation_split=0.1,
              shuffle=True,
              verbose=2)#,
              #class_weight=class_weights)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/msc-hpc/round3/lstm/round3_model1_history1.png', bbox_inches='tight')
# summarize history for loss

plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/msc-hpc/round3/lstm/round3_model1_history2.png', bbox_inches='tight')

model.probs = model.predict_proba(x_test)

import numpy as np
np.savetxt("/home/vljchr004/msc-hpc/round3/lstm/round3_model1_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("/home/vljchr004/msc-hpc/round3/lstm/round3_model1_y_test.csv", np.array(y_test), fmt="%s")

model.save('/home/vljchr004/msc-hpc/round3/lstm/round3_model1_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done------------------------------------------>")

