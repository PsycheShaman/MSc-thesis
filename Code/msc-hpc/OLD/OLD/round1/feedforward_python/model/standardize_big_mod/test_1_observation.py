print("==============================================================================================")

print("starting........................................................................................")

import glob

import numpy as np

print("imported glob, np........................................................................................")

#x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/x_*.pkl")
#y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/y_*.pkl")

x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\ff\\x_*.pkl")
y_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\ff\\y_*.pkl")

import pickle

print("loading first x pickle........................................................................................")

with open(x_files[0], 'rb') as x_file0:
    x = pickle.load(x_file0)
    
print("loading first y pickle........................................................................................")

with open(y_files[0], 'rb') as y_file0:
   y = pickle.load(y_file0)
   
print("recursively adding x pickles........................................................................................")

for i in x_files[1:2]:
#for i in x_files[1:2]:
    with open(i,'rb') as x_file:
        xi = pickle.load(x_file)
        x = np.concatenate((x,xi),axis=0)
        
print("recursively adding y pickles........................................................................................")
        
for i in y_files[1:2]:
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

y = np.concatenate((electrons_y,y,electrons_y),axis=None)

mu = np.mean(x)

sigma = np.std(x)

x = np.subtract(x,mu)

x = np.true_divide(x,sigma)

  
x_add = np.array([np.array((np.sum(i[0:2]),np.sum(i[3:5]),np.sum(i[6:8]),np.sum(i[9:11]),\
                    np.sum(i[12:14]),\
np.sum(i[15:17]),np.sum(i[18:20]),np.sum(i[21:23]))) for i in x])
    
mu = np.mean(x_add)

sigma = np.std(x_add)

x_add = np.subtract(x_add,mu)

x_add = np.true_divide(x_add,sigma)
    
x = np.hstack((x,x_add))

x = x[1:]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x.astype(float))

x = scaler.transform(x.astype(float))

   
from tensorflow.keras.utils import to_categorical

#y = to_categorical(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=123456)

import tensorflow

from tensorflow import keras


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation


num_classes = 2
epochs = 300

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
    
model1_dropout_0_5 = Sequential([
    Dense(1024, input_shape=(32,)),
    Activation('relu'),
    Dropout(0.8),
    Dense(1024),
    Activation('relu'),
    Dropout(0.8),
    Dense(512),
    Activation('relu'),
    Dropout(0.5),
    Dense(512),
    Activation('relu'),
    Dropout(0.5),
    Dense(256),
    Activation('relu'),
    Dropout(0.5),
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(64),
    Activation('relu'),
    Dropout(0.5),
    Dense(2),
    Activation('softmax')
])

model1_dropout_0_5.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


history = model1_dropout_0_5.fit(x_train, y_train,
              #batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
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
plt.savefig('/home/vljchr004/msc-hpc/feedforward_python/fig/feed_forward_1_do_ws_std_history1.png', bbox_inches='tight')
# summarize history for loss

plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/vljchr004/msc-hpc/feedforward_python/fig/feed_forward_1_do_ws_std_history2.png', bbox_inches='tight')

model1_dropout_0_5.probs = model1_dropout_0_5.predict_proba(x_test)

import numpy as np
np.savetxt("/home/vljchr004/msc-hpc/feedforward_python/results/feed_forward_1_do_ws_std_results.csv", np.array(model1_dropout_0_5.probs), fmt="%s")

np.savetxt("/home/vljchr004/msc-hpc/feedforward_python/results/feed_forward_1_do_ws_std_y_test.csv", np.array(y_test), fmt="%s")

model1_dropout_0_5.save('/home/vljchr004/msc-hpc/feedforward_python/feed_forward_1_do_ws_std_.h5')  # creates a HDF5 file 'my_model.h5'
del model1_dropout_0_5 

#model1_dropout_0_8 = Sequential([
#    Dense(256, input_shape=(32,)),
#    Activation('relu'),
#    Dropout(0.8),
#    Dense(128),
#    Activation('relu'),
#    Dropout(0.8),
#    Dense(128),
#    Activation('relu'),
#    Dropout(0.8),
#    Dense(64),
#    Activation('relu'),
#    Dropout(0.8),
#    Dense(2),
#    Activation('softmax')
#])
#
#model1_dropout_0_8.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
#
#
#history = model1_dropout_0_8.fit(x_train, y_train,
#              #batch_size=batch_size,
#              epochs=epochs,
#              validation_split=0.15,
#              shuffle=True,
#              verbose=2)
#
#import matplotlib.pyplot as plt
#
## summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('/home/vljchr004/msc-hpc/feedforward_python/fig/feed_forward_1_dropout_0_8_history.png', bbox_inches='tight')
#
#model1_dropout_0_8.probs = model1_dropout_0_8.predict_proba(x_test)
#
#import numpy as np
#np.savetxt("/home/vljchr004/msc-hpc/feedforward_python/results/feed_forward_1__dropout_0_8_results.csv", np.array(model1_dropout_0_8.probs), fmt="%s")
#
#model1_dropout_0_8.save('/home/vljchr004/msc-hpc/feedforward_python/feed_forward_1__dropout_0_8.h5')  # creates a HDF5 file 'my_model.h5'
#del model1_dropout_0_8
#
#model1_dropout_0_8_0_5 = Sequential([
#    Dense(256, input_shape=(32,)),
#    Activation('relu'),
#    Dropout(0.8),
#    Dense(128),
#    Activation('relu'),
#    Dropout(0.8),
#    Dense(128),
#    Activation('relu'),
#    Dropout(0.5),
#    Dense(64),
#    Activation('relu'),
#    Dropout(0.5),
#    Dense(2),
#    Activation('softmax')
#])
#
#model1_dropout_0_8_0_5.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
#
#
#history = model1_dropout_0_8_0_5.fit(x_train, y_train,
#              #batch_size=batch_size,
#              epochs=epochs,
#              validation_split=0.15,
#              shuffle=True,
#              verbose=2)
#
#import matplotlib.pyplot as plt
#
## summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('/home/vljchr004/msc-hpc/feedforward_python/fig/feed_forward_1_dropout_0_8_0_5_history2.png', bbox_inches='tight')
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('/home/vljchr004/msc-hpc/feedforward_python/fig/feed_forward_1_dropout_0_8_0_5_history2.png', bbox_inches='tight')
#
#model1_dropout_0_8_0_5.probs = model1_dropout_0_8_0_5.predict_proba(x_test)
#
#import numpy as np
#np.savetxt("/home/vljchr004/msc-hpc/feedforward_python/results/feed_forward_1__dropout_0_8_0_5_results.csv", np.array(model1_dropout_0_8_0_5.probs), fmt="%s")
#
#model1_dropout_0_8_0_5.save('/home/vljchr004/msc-hpc/feedforward_python/feed_forward_1__dropout_0_8_0_5.h5')  # creates a HDF5 file 'my_model.h5'
#del model1_dropout_0_8_0_5
#
