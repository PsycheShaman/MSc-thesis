# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:07:00 2019

@author: https://keras.io/examples/variational_autoencoder/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt


import pickle 
import glob

from keras.optimizers import Adam

def load_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        
        with open(x_files[0],'rb') as x_file:
            x = pickle.load(x_file)
        
        for i in x_files[1:8]:
            print(i)
            with open(i,'rb') as x_file:
                print(i)
                xi = pickle.load(x_file)
                x = np.concatenate((x,xi),axis=0)
                print(x.shape)
        return(x)
        
def scale(x, out_range=(0, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def sampling(args):

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def sample_images(models,data,epoch, latent_dim=5,batch_size=500000):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_imgs = decoder.predict(noise)
#    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs = (gen_imgs-np.min(gen_imgs))/(np.max(gen_imgs)-np.min(gen_imgs))
    gen_imgs = gen_imgs.reshape(-1,17,24,1)
    # Rescale images 0 - 1
    for i in range(batch_size):
        
        plt.imshow(gen_imgs[i,:,:,0],cmap='gray')
        plt.savefig("images/vae"+str(epoch)+"_"+str(i)+".png")
        plt.close()

x_train = load_data()

original_dim = 17 * 24
x_train = np.reshape(x_train, [-1, original_dim])
x_train = scale(x_train)

input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 5
epochs = 50

from keras.layers import BatchNormalization, LeakyReLU

# VAE model = encoder + decode3r
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim)(inputs)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(intermediate_dim)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(intermediate_dim)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(intermediate_dim)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(intermediate_dim)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(intermediate_dim)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim)(latent_inputs)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(intermediate_dim)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(intermediate_dim)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(intermediate_dim)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(intermediate_dim)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(intermediate_dim)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

models = (encoder, decoder)
data = x_train


    
from keras import backend as K
import tensorflow as tf

import dill

def categorical_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """

    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = 2 * K.pow(1 - y_pred, .25) * cross_entropy

    # Sum the losses in mini_batch
    return K.sum(loss, axis=1)


reconstruction_loss = categorical_focal_loss_fixed(inputs, outputs)
#
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)



vae.add_loss(vae_loss)

#import keras.optimizers.Adam

adam = Adam(lr=0.00001)

vae.load_weights("C:/Users/gerhard/documents/msc-thesis/code/latent variable models/newvae/m13/weights_checkpoint/vae_weights_epoch_48_loss_0-5163.h5")

vae.compile(optimizer=adam)
vae.summary()
plot_model(vae,
           to_file='vae_mlp.png',
           show_shapes=True)


        
def sample_images2(models,j):
#        r, c = 5, 5
        noise = np.random.normal(0, 1, (200, latent_dim))
        gen_imgs = decoder.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = (gen_imgs-np.min(gen_imgs))/(np.max(gen_imgs)-np.min(gen_imgs))
        gen_imgs = gen_imgs.reshape(-1,17,24,1)
        np.save("simulated_data/vae"+str(j)+".npy",arr=gen_imgs)
        
batch_size=256
        
for j in range(1000000000000000000000000000000000000000000000000000):
    print("macro_epoch_batch_"+str(j))
    vae.fit(x_train[np.random.randint(low=0,high=x_train.shape[0],size=1000000),:],
                epochs=1,
                batch_size=batch_size)
#        ,
#                validation_data=(x_test, None))
    vae.save_weights('vae_mlp_mnist.h5')

    sample_images(models,
                 data,
                 batch_size=3,
                 epoch=j)
    sample_images2(models,j)
    
adam = Adam(lr=1e-9)
vae.compile(optimizer=adam)

for j in range(3,1000000000000000000000000000000000000000000000000000):
    print("macro_epoch_batch_"+str(j))
    vae.fit(x_train[np.random.randint(low=0,high=x_train.shape[0],size=1000000),:],
                epochs=1,
                batch_size=batch_size)
#        ,
#                validation_data=(x_test, None))
    vae.save_weights('vae_mlp_mnist.h5')

    sample_images(models,
                 data,
                 batch_size=3,
                 epoch=j)
    sample_images2(models,j)
    
#THE ABOVE DOES NOT WORK, GOING BACK TO LOADING WHEIGHTS AND LARGER LR

adam = Adam(lr=0.00001)

vae.load_weights("C:/Users/gerhard/documents/msc-thesis/code/latent variable models/newvae/m13/weights_checkpoint/vae_weights_epoch_48_loss_0-5163.h5")

vae.compile(optimizer=adam)

for j in range(5,1000000000000000000000000000000000000000000000000000):
    print("macro_epoch_batch_"+str(j))
    vae.fit(x_train[np.random.randint(low=0,high=x_train.shape[0],size=50000),:],
                epochs=1,
                batch_size=batch_size)
#        ,
#                validation_data=(x_test, None))
    vae.save_weights('vae_mlp_mnist.h5')

    sample_images(models,
                 data,
                 batch_size=3,
                 epoch=j)
    sample_images2(models,j)
    

# remains stuck at around 6770, reducing LR
    
adam = Adam(lr=0.000001)

vae.compile(optimizer=adam)

for j in range(1302,1000000000000000000000000000000000000000000000000000):
    print("macro_epoch_batch_"+str(j))
    vae.fit(x_train[np.random.randint(low=0,high=x_train.shape[0],size=50000),:],
                epochs=1,
                batch_size=batch_size)
#        ,
#                validation_data=(x_test, None))
    vae.save_weights('vae_mlp_mnist.h5')

    sample_images(models,
                 data,
                 batch_size=3,
                 epoch=j)
    sample_images2(models,j)
    
# loss is not really decreasing anymore

# attempting to plot latent space representation

import os

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test = data
    os.makedirs(model_name, exist_ok=True)


    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
#    digit_size = 28
    figure = np.zeros((17 * n, 24 * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[0,0,0,xi,yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(17, 24)
            figure[i * 17: (i + 1) * 17,
                   j * 24: (j + 1) * 24] = digit

    plt.figure(figsize=(10, 10))
#    start_range = 17 // 2
#    end_range = (n - 1) * 1 + start_range + 1
#    pixel_range = np.arange(start_range, end_range, 24)
#    sample_range_x = np.round(grid_x, 1)
#    sample_range_y = np.round(grid_y, 1)
    plt.xticks((0,175,350,525,700),(-4,-2,0,2,4))
    plt.yticks((0,125,250,375,500), (-4,-2,0,2,4))
    plt.xlabel("z[3]")
    plt.ylabel("z[4]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


plot_results(models,data)

#Discrimination

noise = np.random.normal(0, 1, (x_train.shape[0], latent_dim))
gen_imgs = decoder.predict(noise)

def scale(x, out_range=(0, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

        # Rescale images 0 - 1
gen_imgs = scale(gen_imgs) #(gen_imgs-np.min(gen_imgs))/(np.max(gen_imgs)-np.min(gen_imgs))
gen_imgs = gen_imgs.reshape(-1,17,24,1)

x_train = x_train.reshape(-1,17,24,1)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(4,4),strides=1, padding='valid',activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(4,4),strides=1, padding='valid',activation="relu"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation="sigmoid"))
model.add(tf.keras.layers.Dense(256,activation="sigmoid"))
model.add(tf.keras.layers.Dense(128,activation="sigmoid"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.00001),
              metrics=['accuracy'])

batch_size=32

epochs=30

x = np.concatenate((gen_imgs,x_train),axis=0)
y = np.concatenate((np.zeros(gen_imgs.shape[0]),np.ones(x_train.shape[0])))
    
history=model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs)


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/final_vae_vs_real__history1.png', bbox_inches='tight')
plt.close()


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('C:/Users/gerhard/Documents/MSc-thesis/final_vae_vs_real__history2.png', bbox_inches='tight')

plt.close()


vae_preds = model.predict(gen_imgs)
plt.hist(vae_preds)

np.savetxt("vae_preds.csv",X=vae_preds)
 
real_preds = model.predict(x_train)
np.savetxt("real_preds.csv",X=real_preds)

plt.hist(real_preds)

x_train = load_data()

def scale(x, out_range=(0, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

gen_imgs = scale(gen_imgs,(np.min(x_train),np.max(x_train)))

i = np.where((vae_preds>0.1) & (vae_preds<=0.2))[0][0]
i = np.where(vae_preds==np.max(vae_preds))[0][1]
p_real = vae_preds[i][0]
plt.imshow(gen_imgs[i,:,:,0],cmap='gray')
plt.colorbar()
plt.title("P(real)= "+str(p_real))

#plt.hist(geant_preds,bins=10,range=(0,1))
#plt.title('Histogram of P(real) Predictions for Fully Connected VAE Data')
#plt.ylabel('Frequency')
#plt.xlabel('P(real)')
#
#real_preds = model.predict(real)
#
#plt.hist(real_preds,bins=10,range=(0,1))
#plt.title('Histogram of P(real) Predictions for Real Data')
#plt.ylabel('Frequency')
#plt.xlabel('P(real)')

x_train = scale(x_train,(0,1))

#def plot_results(models,
#                 data,
#                 batch_size=128,
#                 model_name="vae_mnist"):
#    """Plots labels and MNIST digits as a function of the 2D latent vector
#
#    # Arguments
#        models (tuple): encoder and decoder models
#        data (tuple): test data and label
#        batch_size (int): prediction batch size
#        model_name (string): which model is using this function
#    """
#
#    encoder, decoder = models
#    x_test, y_test = data
#    os.makedirs(model_name, exist_ok=True)
#    x_test = x_test.reshape(-1,17*24)
#    filename = "vae_mean.png"
#    # display a 2D plot of the digit classes in the latent space

gen_imgs = scale(gen_imgs,(0,1))
gen_imgs = gen_imgs.reshape(-1,17*24)
s = np.random.randint(0,gen_imgs.shape[0],100)
gen_imgs2 = gen_imgs[s,:]
    
z_mean, _, _ = encoder.predict(gen_imgs,
                               batch_size=512)

import seaborn as sns

cmap = sns.cubehelix_palette(as_cmap=True)

#f, ax = plt.subplots()
#points = ax.scatter(x=z_mean[:, 0], y=z_mean[:, 1], c=vae_preds[s], s=50, cmap=cmap)
#f.colorbar(points)

s = vae_preds

s.shape = s.shape[0]

import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
import matplotlib.ticker

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

i=1
#j=1
for j in range(2,4):
    if(i==j):continue
    plt.figure(figsize=(6,5))
    
    plt.scatter(x=noise[(s>=0.99), i], y=noise[(s>=0.99), j],
                c=s[(s>=0.99)],cmap=cmap,s=50,alpha=0.3)
    plt.colorbar()
    plt.xlabel("z["+str(i)+"]")
    plt.ylabel("z["+str(j)+"]")
    plt.show()
    plt.savefig("vae_high"+str(i)+"_"+str(j)+".png")
    
    plt.figure(figsize=(6,5))
    
    plt.scatter(x=noise[(s>=0.49)&(s<=0.51), i], y=noise[(s>=0.49)&(s<=0.51), j],
                c=s[(s>=0.49)&(s<=0.51)],cmap=cmap,s=50,alpha=0.3)
    plt.colorbar()
    plt.xlabel("z["+str(i)+"]")
    plt.ylabel("z["+str(j)+"]")
    plt.show()
    plt.savefig("vae_med"+str(i)+"_"+str(j)+".png")
    
    n=1000
    plt.figure(figsize=(6,5))
    fig, ax = plt.subplots()
    sipsx = noise[(s<=0.000001), i]
    sipsx = sipsx[0:n]
    sipsy = noise[(s<=0.000001), j]
    sipsy = sipsy[0:n]
    sipsc = s[(s<=0.000001)]
    sipsc = sipsc[0:n]
    plot = ax.scatter(x=sipsx, y=sipsy,
                c=sipsc,cmap=cmap,s=50,alpha=0.3)
    
    cbar = fig.colorbar(plot,format=OOMFormatter(-7, mathText=True))
    plt.xlabel("z["+str(i)+"]")
    plt.ylabel("z["+str(j)+"]")
    plt.show()
    plt.savefig("vae_low"+str(i)+"_"+str(j)+".png")

#plot_results(models,data=(x_train,real_preds))
#
#
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#
#x, y, z = np.random.rand(3, 100)
#cmap = sns.cubehelix_palette(as_cmap=True)
#
#f, ax = plt.subplots()
#points = ax.scatter(x, y, c=z, s=50, cmap=cmap)
#f.colorbar(points)


    
