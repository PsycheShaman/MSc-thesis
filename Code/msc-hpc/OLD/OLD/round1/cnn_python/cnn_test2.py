# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:01:37 2019

@author: Gerhard
"""

print("==============================================================================================")

import pickle

import numpy as np
 
#with open('/scratch/vljchr004/data/msc-thesis-data/cnn/x.pkl', 'rb') as x_file:
#    x = pickle.load(x_file)
#
#with open('/scratch/vljchr004/data/msc-thesis-data/cnn/y.pkl', 'rb') as y_file:
#    y = pickle.load(y_file)

with open('C:/Users/gerhard/Documents/msc-thesis-data/ff/x_000265309.pkl', 'rb') as x_file:
    x = pickle.load(x_file)

with open('C:/Users/gerhard/Documents/msc-thesis-data/ff/y_000265309.pkl', 'rb') as y_file:
    y = pickle.load(y_file)
    
#x = x.reshape(x.shape[0],x.shape[1],1,1)

x = [np.diag(i) for i in x]

x = np.stack(x)

x.shape = (x.shape[0],x.shape[1],x.shape[2],1)

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
                 input_shape=(24,24,1)))
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
#plt.savefig('/home/vljchr004/msc-hpc/cnn_python/fig/cnn_1_history1.png', bbox_inches='tight')

plt.savefig('C:/Users/gerhard/Documents/msc-hpc/cnn_python/fig/test/cnn_2_history1.png', bbox_inches='tight')


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('/home/vljchr004/msc-hpc/cnn_python/fig/cnn_1_history2.png', bbox_inches='tight')

plt.savefig('C:/Users/gerhard/Documents/msc-hpc/cnn_python/fig/test/cnn_2_history2.png', bbox_inches='tight')

model.probs = model.predict_proba(x_test)

import numpy as np
#np.savetxt("/home/vljchr004/msc-hpc/cnn_python/results/cnn_1_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/msc-hpc/cnn_python/results/test/cnn_2_results.csv", np.array(model.probs), fmt="%s")

#np.savetxt("/home/vljchr004/msc-hpc/cnn_python/results/cnn_1_y_test.csv", np.array(y_test), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/msc-hpc/cnn_python/results/test/cnn_2_y_test.csv", np.array(y_test), fmt="%s")

#model.save('/home/vljchr004/msc-hpc/cnn_python/cnn1.h5')  # creates a HDF5 file 'my_model.h5'

model.save('C:/Users/gerhard/Documents/msc-hpc/cnn_python/results/test/cnn2.h5')

del model


















