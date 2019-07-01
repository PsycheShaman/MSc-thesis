# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:26:58 2019

@author: gerhard
"""

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1024,activation="sigmoid",input_shape=(35,)))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1024,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(512,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(512,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(256,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(256,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(64,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(64,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model48.png',show_shapes=True)


import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1024,activation="sigmoid",input_shape=(35,)))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(512,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(256,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(64,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model47.png',show_shapes=True)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(1000,return_sequences=False,input_shape=(24,17)))
model.add(tf.keras.layers.Dense(1000,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model46.png',show_shapes=True)


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(512,return_sequences=False,input_shape=(24,17)))
model.add(tf.keras.layers.Dense(512,activation="tanh"))
model.add(tf.keras.layers.Dense(512,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(64,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model45.png',show_shapes=True)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(256,return_sequences=True,input_shape=(24,17)))
model.add(tf.keras.layers.LSTM(256,return_sequences=False))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model44.png',show_shapes=True)


import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128,return_sequences=True,input_shape=(24,17)))
model.add(tf.keras.layers.LSTM(128,return_sequences=False))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model43.png',show_shapes=True)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32,kernel_size=6,input_shape=(24,1)))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Conv1D(filters=64,kernel_size=3))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Conv1D(filters=128,kernel_size=2))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation="tanh"))
model.add(tf.keras.layers.Dense(512,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(64,activation="tanh"))
model.add(tf.keras.layers.Dense(32,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model42.png',show_shapes=True)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=6,kernel_size=2, input_shape=(24,1)))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Conv1D(filters=12,kernel_size=2))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Conv1D(filters=24,kernel_size=2))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Conv1D(filters=48,kernel_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation="tanh"))
model.add(tf.keras.layers.Dense(512,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(64,activation="tanh"))
model.add(tf.keras.layers.Dense(32,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model41.png',show_shapes=True)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=6,kernel_size=2, input_shape=(24,1)))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Conv1D(filters=12,kernel_size=2))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Conv1D(filters=24,kernel_size=2))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model40.png',show_shapes=True)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=6,kernel_size=2, input_shape=(24,1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model39.png',show_shapes=True)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=6,kernel_size=2, input_shape=(24,1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model38.png',show_shapes=True)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(256,return_sequences=True,go_backwards=False,input_shape=(24,17)))
model.add(tf.keras.layers.LSTM(256,return_sequences=True,go_backwards=True))
model.add(tf.keras.layers.LSTM(256,return_sequences=True,go_backwards=False))
model.add(tf.keras.layers.LSTM(256,return_sequences=True,go_backwards=True))
model.add(tf.keras.layers.LSTM(256,return_sequences=True,go_backwards=False))
model.add(tf.keras.layers.LSTM(256,return_sequences=False,go_backwards=True))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/NEW/plotting/model37.png',show_shapes=True)



























model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(17,24),strides=1, padding='valid',activation="relu",input_shape=(17,24,1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(12,activation="sigmoid"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='C:/Users/Gerhard/Documents/MSc-thesis/plotting/model1.png',show_shapes=True)


