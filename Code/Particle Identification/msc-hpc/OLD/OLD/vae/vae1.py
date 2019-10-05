print("==============================================================================================")

print("starting........................................................................................")

import glob

import numpy as np

print("imported glob, np........................................................................................")

#x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/x_*.pkl")
#y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/y_*.pkl")

x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")

import pickle

print("loading first x pickle........................................................................................")

with open(x_files[0], 'rb') as x_file0:
    x = pickle.load(x_file0)
   
print("recursively adding x pickles........................................................................................")

for i in x_files[1:]:
    with open(i,'rb') as x_file:
        print(i)
        xi = pickle.load(x_file)
        x = np.concatenate((x,xi),axis=0)
 
    
#x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.npy")
#
##x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/x_*.npy")
##y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/y_*.npy")
#       
#print("recursively adding x numpys........................................................................................")
#
#for i in x_files[0:]:
#    with open(i,'rb') as x_file:
#        print(i)
#        xi = np.load(x_file)
#        x = np.concatenate((x,xi),axis=0)       

print("removing useless elements........................................................................................")

nz = np.array([np.count_nonzero(i) for i in x])

zeros = np.where(nz==0)

x = np.delete(x,zeros,axis=0)

x.shape = (x.shape[0],408)

mu = np.mean(x)

x = x.astype('float32')/mu

from sklearn.model_selection import train_test_split

x_train, x_test = train_test_split(x, test_size=0.2,random_state=123456)



import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(408,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(408, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from tensorflow.python.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/autoencoder1")

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=2,
                callbacks=[tensorboard])

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(17, 24))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(17, 24))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

























