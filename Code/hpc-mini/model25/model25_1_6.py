import numpy as np
import matplotlib.pyplot as plt

tracks = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("/scratch/vljchr004/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24, 1))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

import tensorflow as tf

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2DTranspose(filters=16,kernel_size=(3,5),strides=1, padding='valid',activation="relu",input_shape=(17,24,1)))
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,5),strides=1, padding='valid',activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(256,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

adam = tf.train.AdamOptimizer(learning_rate=0.000001)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=100

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
plt.savefig('/home/vljchr004/hpc-mini/chamber_gain_corrected/model25_1_6_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/hpc-mini/chamber_gain_corrected/model25_1_6_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(train)

import numpy as np
np.savetxt("/home/vljchr004/hpc-mini/chamber_gain_corrected/model25_1_6_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("/home/vljchr004/hpc-mini/chamber_gain_corrected/model25_1_6_y_test.csv", np.array(labels), fmt="%s")
print("saving model")
model.save('/home/vljchr004/hpc-mini/chamber_gain_corrected/model25_1_6.h5')  # creates a HDF5 file 'my_model.h5'
print("model saved")

print("<-----------------------------done------------------------------------------>")
