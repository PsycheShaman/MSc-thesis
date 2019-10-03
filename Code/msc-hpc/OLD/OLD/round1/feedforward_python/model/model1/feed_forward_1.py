print("==============================================================================================")

print("starting........................................................................................")

import glob

import numpy as np

print("imported glob, np........................................................................................")

x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/x_*.pkl")
y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/y_*.pkl")

#x_files = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/ff/x_*.pkl")
#y_files = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/ff/y_*.pkl")

import pickle

print("loading first x pickle........................................................................................")

with open(x_files[0], 'rb') as x_file0:
    x = pickle.load(x_file0)
    
print("loading first y pickle........................................................................................")

with open(y_files[0], 'rb') as y_file0:
   y = pickle.load(y_file0)
   
print("recursively adding x pickles........................................................................................")

for i in x_files[1:]:
#for i in x_files[1:2]:
    with open(i,'rb') as x_file:
        xi = pickle.load(x_file)
        x = np.concatenate((x,xi),axis=0)
        
print("recursively adding y pickles........................................................................................")
        
for i in y_files[1:]:
#for i in y_files[1:2]:
    with open(i,'rb') as y_file:
        yi = pickle.load(y_file)
        y = np.concatenate((y,yi),axis=None)
        
nz = np.array([np.count_nonzero(i) for i in x])

zeros = np.where(nz==0)

x = np.delete(x,zeros,axis=0)
y = np.delete(y,zeros)

#oversample electrons

elec = np.where(y==1)
pion = np.where(y!=1)

electrons_x = x[elec,:]

electrons_y = y[elec]

electrons_x = np.squeeze(electrons_x)

x = np.concatenate((electrons_x,x,electrons_x),axis=0)

mu = np.mean(x)

x = np.true_divide(x,mu)

y = np.concatenate((electrons_y,y,electrons_y),axis=None)
    
from tensorflow.keras.utils import to_categorical

#y = to_categorical(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=123456)

import tensorflow

from tensorflow import keras


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation


num_classes = 2
epochs = 100

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
    
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

model1.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


history = model1.fit(x_train, y_train,
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
plt.savefig('/home/vljchr004/msc-hpc/feedforward_python/fig/feed_forward_1_history1_normalized.png', bbox_inches='tight')

plt.close()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/msc-hpc/feedforward_python/fig/feed_forward_1_history2_normalized.png', bbox_inches='tight')

model1.probs = model1.predict_proba(x_test)

import numpy as np
np.savetxt("/home/vljchr004/msc-hpc/feedforward_python/results/feed_forward_1_results_normalized.csv", np.array(model1.probs), fmt="%s")

np.savetxt("/home/vljchr004/msc-hpc/feedforward_python/results/feed_forward_1_y_test_normalized.csv", np.array(y_test), fmt="%s")

model1.save('/home/vljchr004/msc-hpc/feedforward_python/feed_forward1_normalized.h5')  # creates a HDF5 file 'my_model.h5'

del model1 




