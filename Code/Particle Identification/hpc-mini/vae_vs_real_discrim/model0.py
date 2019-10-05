import numpy as np
import matplotlib.pyplot as plt

real = np.load("C:/Users/gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")

real = real.reshape((-1, 17, 24))

for i in range(0,real.shape[0]):
    ma = np.max(real[i,:,:])
    real[i,:,:] = real[i,:,:]/ma
    
import glob

files_list = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/generated-data/vae11/*.npy")

simulated = np.load(files_list[0]) 

for i in files_list[1:]:
    with open(i,'rb') as x_file:
        print(i)
        xi = np.load(x_file)
        simulated = np.concatenate((simulated,xi),axis=0) 
        
for i in range(0,simulated.shape[0]):
    ma = np.max(simulated[i,:,:])
    simulated[i,:,:] = simulated[i,:,:]/ma

real = real[0:100000,:,:]

real_y = np.repeat(1,100000)

simulated_y = np.repeat(0,100000)

x = np.vstack((real,simulated))

y = np.concatenate((real_y,simulated_y))

x.shape = (x.shape[0],x.shape[1],x.shape[2],1)


import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(8,12),strides=1, padding='valid',activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),strides=1, padding='valid',activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=1, padding='valid',activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

adam = tf.train.AdamOptimizer(learning_rate=0.00001) 

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size=32

epochs=5
    
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
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/vae_vs_real_discrim/model0_history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/gerhard/Documents/hpc-mini/vae_vs_real_discrim/model0_history2.png', bbox_inches='tight')

plt.close()

model.probs = model.predict_proba(x)

import numpy as np
np.savetxt("C:/Users/gerhard/Documents/hpc-mini/vae_vs_real_discrim/model0_results.csv", np.array(model.probs), fmt="%s")

np.savetxt("C:/Users/gerhard/Documents/hpc-mini/vae_vs_real_discrim/model0_y_test.csv", np.array(y), fmt="%s")

model.save('C:/Users/gerhard/Documents/hpc-mini/vae_vs_real_discrim/model0_.h5')  # creates a HDF5 file 'my_model.h5'
del model

print("<-----------------------------done------------------------------------------>")





















