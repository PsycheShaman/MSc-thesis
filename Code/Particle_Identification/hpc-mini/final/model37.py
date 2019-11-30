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
plt.savefig('/home/vljchr004/hpc-mini/chamber_gain_corrected/model37_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/hpc-mini/chamber_gain_corrected/model37_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(train)

import numpy as np
np.savetxt("/home/vljchr004/hpc-mini/chamber_gain_corrected/model37_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("/home/vljchr004/hpc-mini/chamber_gain_corrected/model37_y_test.csv", np.array(labels), fmt="%s")

model.save('/home/vljchr004/hpc-mini/chamber_gain_corrected/model37_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done------------------------------------------>")
