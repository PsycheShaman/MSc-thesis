# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:16:21 2019

@author: gerhard
"""

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("c:/Users/gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("c:/Users/gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 24, 17))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(512,return_sequences=True))
model.add(tf.keras.layers.LSTM(256,return_sequences=True))
model.add(tf.keras.layers.LSTM(128,return_sequences=True))
model.add(tf.keras.layers.LSTM(64,return_sequences=True))
model.add(tf.keras.layers.LSTM(32,return_sequences=False))
model.add(tf.keras.layers.Dense(32,activation="tanh"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=50

history=model.fit(train, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model34_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model34_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(train)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model34_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model34_y_test.csv", np.array(labels), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model34_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:34------------------------------------------>")

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 24, 17))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

import tensorflow as tf

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

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=50

history=model.fit(train, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model37_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model37_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(train)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model37_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model37_y_test.csv", np.array(labels), fmt="%s")

del model

print("<-----------------------------done:37------------------------------------------>")

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

x  = []

for i in range(0,train.shape[0]):
    xi = train[i,:,:].sum(axis=0)
    x.append(xi)

x = np.array(x)

x.shape = (x.shape[0],x.shape[1],1)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=6,kernel_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=100

history=model.fit(x, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model38_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model38_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model38_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model38_y_test.csv", np.array(labels), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model38_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:38------------------------------------------>")

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

x  = []

for i in range(0,train.shape[0]):
    xi = train[i,:,:].sum(axis=0)
    x.append(xi)

x = np.array(x)

x.shape = (x.shape[0],x.shape[1],1)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=6,kernel_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=100

history=model.fit(x, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model39_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model39_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model39_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model39_y_test.csv", np.array(labels), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model39_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:39------------------------------------------>")

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

x  = []

for i in range(0,train.shape[0]):
    xi = train[i,:,:].sum(axis=0)
    x.append(xi)

x = np.array(x)

x.shape = (x.shape[0],x.shape[1],1)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=6,kernel_size=2))
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

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=100

history=model.fit(x, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model40_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model40_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model40_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model40_y_test.csv", np.array(labels), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model40_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:40------------------------------------------>")


import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

x  = []

for i in range(0,train.shape[0]):
    xi = train[i,:,:].sum(axis=0)
    x.append(xi)

x = np.array(x)

x.shape = (x.shape[0],x.shape[1],1)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=6,kernel_size=2))
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

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=100

history=model.fit(x, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model41_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model41_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model41_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model41_y_test.csv", np.array(labels), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model41_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:41------------------------------------------>")

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

x  = []

for i in range(0,train.shape[0]):
    xi = train[i,:,:].sum(axis=0)
    x.append(xi)

x = np.array(x)

x.shape = (x.shape[0],x.shape[1],1)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32,kernel_size=6))
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

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=100

history=model.fit(x, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model42_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model42_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model42_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model42_y_test.csv", np.array(labels), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model42_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:42------------------------------------------>")


import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

x  = []

for i in range(0,train.shape[0]):
    xi = train[i,:,:].sum(axis=0)
    x.append(xi)

x = np.array(x)

x.shape = (x.shape[0],x.shape[1],1)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128,return_sequences=True))
model.add(tf.keras.layers.LSTM(128,return_sequences=False))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=100

history=model.fit(x, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model43_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model43_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model43_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model43_y_test.csv", np.array(labels), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model43_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:43------------------------------------------>")

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

x  = []

for i in range(0,train.shape[0]):
    xi = train[i,:,:].sum(axis=0)
    x.append(xi)

x = np.array(x)

x.shape = (x.shape[0],x.shape[1],1)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(256,return_sequences=True))
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

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=100

history=model.fit(x, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model44_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model44_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model44_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model44_y_test.csv", np.array(labels), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model44_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:44------------------------------------------>")

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

x  = []

for i in range(0,train.shape[0]):
    xi = train[i,:,:].sum(axis=0)
    x.append(xi)

x = np.array(x)

x.shape = (x.shape[0],x.shape[1],1)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(512,return_sequences=False))
model.add(tf.keras.layers.Dense(512,activation="tanh"))
model.add(tf.keras.layers.Dense(512,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(256,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(64,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=100

history=model.fit(x, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model45_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model45_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model45_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model45_y_test.csv", np.array(labels), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model45_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:45------------------------------------------>")

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

x  = []

for i in range(0,train.shape[0]):
    xi = train[i,:,:].sum(axis=0)
    x.append(xi)

x = np.array(x)

x.shape = (x.shape[0],x.shape[1],1)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(1000,return_sequences=False,input_shape=(24,17)))
model.add(tf.keras.layers.Dense(1000,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=20

history=model.fit(x, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model46_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model46_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model46_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model46_y_test.csv", np.array(labels), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model46_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:46------------------------------------------>")

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24))

y = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

#train = train/ma
#
x  = []

for i in range(0,train.shape[0]):
    xi = train[i,:,:].sum(axis=0)
    x.append(xi)

x = np.array(x)

x.shape = (x.shape[0],x.shape[1])

mu = []

for i in range(0,x.shape[0]):
    mu.append(np.mean(x[i,:]))

mu = np.array(mu)

sd = []

for i in range(0,x.shape[0]):
    sd.append(np.std(x[i,:]))

sd = np.array(sd)

l2 = []

for i in range(0,x.shape[0]):
    l2.append(np.linalg.norm(x[i,:],2))

l2 = np.array(l2)


ma = []

for i in range(0,x.shape[0]):
    ma.append(np.max(x[i,:]))

ma = np.array(ma)

gr1 = []

for i in range(0,x.shape[0]):
    gr1.append(np.sum(x[i,0:5]))

gr1 = np.array(gr1)

gr2 = []

for i in range(0,x.shape[0]):
    gr2.append(np.sum(x[i,6:11]))

gr2 = np.array(gr2)

gr3 = []

for i in range(0,x.shape[0]):
    gr3.append(np.sum(x[i,12:17]))

gr3 = np.array(gr3)

gr4 = []

for i in range(0,x.shape[0]):
    gr4.append(np.sum(x[i,18:23]))

gr4 = np.array(gr4)

diff1 = gr2-gr1

diff2 = gr3-gr2

diff3 = gr4-gr3

x_add = np.vstack((ma,mu,sd,l2,gr1,gr2,gr3,gr4,diff1,diff2,diff3))

x_add = np.transpose(x_add)

x = np.hstack((x,x_add))

x = x / x.max(axis=0)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1024,activation="sigmoid"))
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

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=100

history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model47_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model47_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model47_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model47_y_test.csv", np.array(y), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model47_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:47------------------------------------------>")

import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

#tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
#
#infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24))

y = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

#train = train/ma
#
x  = []

for i in range(0,train.shape[0]):
    xi = train[i,:,:].sum(axis=0)
    x.append(xi)

x = np.array(x)

x.shape = (x.shape[0],x.shape[1])

mu = []

for i in range(0,x.shape[0]):
    mu.append(np.mean(x[i,:]))

mu = np.array(mu)

sd = []

for i in range(0,x.shape[0]):
    sd.append(np.std(x[i,:]))

sd = np.array(sd)

l2 = []

for i in range(0,x.shape[0]):
    l2.append(np.linalg.norm(x[i,:],2))

l2 = np.array(l2)


ma = []

for i in range(0,x.shape[0]):
    ma.append(np.max(x[i,:]))

ma = np.array(ma)

gr1 = []

for i in range(0,x.shape[0]):
    gr1.append(np.sum(x[i,0:5]))

gr1 = np.array(gr1)

gr2 = []

for i in range(0,x.shape[0]):
    gr2.append(np.sum(x[i,6:11]))

gr2 = np.array(gr2)

gr3 = []

for i in range(0,x.shape[0]):
    gr3.append(np.sum(x[i,12:17]))

gr3 = np.array(gr3)

gr4 = []

for i in range(0,x.shape[0]):
    gr4.append(np.sum(x[i,18:23]))

gr4 = np.array(gr4)

diff1 = gr2-gr1

diff2 = gr3-gr2

diff3 = gr4-gr3

x_add = np.vstack((ma,mu,sd,l2,gr1,gr2,gr3,gr4,diff1,diff2,diff3))

x_add = np.transpose(x_add)

x = np.hstack((x,x_add))

x = x / x.max(axis=0)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1024,activation="sigmoid"))
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

adam = tf.train.AdamOptimizer(learning_rate=0.00001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=100

history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)

#unit = model.get_weights()[0]
#
#unit = model.get_weights()[0][:,:,0,0]
#
#plt.imshow(unit)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model48_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model48_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model48_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model48_y_test.csv", np.array(y), fmt="%s")

#model.save('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model48_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done:48------------------------------------------>")



































































