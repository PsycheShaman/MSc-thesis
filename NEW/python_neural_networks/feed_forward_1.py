# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:47:42 2019

@author: gerhard
"""

import pickle
 
with open('x.full', 'rb') as x_file:
    x = pickle.load(x_file)

with open('y.full', 'rb') as y_file:
    y = pickle.load(y_file)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D

#batch_size = 32
num_classes = 2
epochs = 100

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#import os



#save_dir = os.path.join(os.getcwd(), 'new_models')
#model_name = 'keras_cifar10_trained_model.h5'

#with tf.device('/device:GPU:0'):
    
model1 = Sequential([
    Dense(256, input_shape=(24,)),
    Activation('relu'),
    Dense(128),
    Activation('relu'),
    Dense(128),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(2),
    Activation('softmax')
])
    
    # initiate RMSprop optimizer
    #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    
    # Let's train the model using RMSprop
model1.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


history = model1.fit(x_train, y_train,
              #batch_size=batch_size,
              epochs=epochs,
              validation_split=0.15,
              shuffle=True)

model1.probs = model1.predict_proba(x_test)