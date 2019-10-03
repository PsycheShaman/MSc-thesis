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
plt.savefig('/home/vljchr004/hpc-mini/chamber_gain_corrected/model48_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/hpc-mini/chamber_gain_corrected/model48_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("/home/vljchr004/hpc-mini/chamber_gain_corrected/model48_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("/home/vljchr004/hpc-mini/chamber_gain_corrected/model48_y_test.csv", np.array(y), fmt="%s")

model.save('/home/vljchr004/hpc-mini/chamber_gain_corrected/model48_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done------------------------------------------>")
