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
















    
