# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:07:00 2019

based on: https://keras.io/examples/variational_autoencoder/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
#from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
#import argparse
#import os

import pickle 
import glob

from keras.optimizers import Adam

def load_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        
        with open(x_files[0],'rb') as x_file:
            x = pickle.load(x_file)
        
#        for i in x_files[1:]:
#            print(i)
#            with open(i,'rb') as x_file:
#                print(i)
#                xi = pickle.load(x_file)
#                x = np.concatenate((x,xi),axis=0)
#                print(x.shape)
        return(x)
        
def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def sample_images(models,data,epoch, latent_dim=4,batch_size=500000):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_imgs = decoder.predict(noise)
#    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs = (gen_imgs-np.min(gen_imgs))/(np.max(gen_imgs)-np.min(gen_imgs))
    gen_imgs = gen_imgs.reshape(-1,17,24,1)
    # Rescale images 0 - 1
    for i in range(batch_size):
        
        plt.imshow(gen_imgs[i,:,:,0],cmap='gray')
        plt.savefig("images/vae"+str(epoch)+".png")
        plt.close()



x_train = load_data()
#x_test = load_data()

#image_size = x_train.shape[1]
original_dim = 17 * 24
x_train = np.reshape(x_train, [-1, original_dim])
#x_test = np.reshape(x_test, [-1, original_dim])
x_train = scale(x_train)
#x_test = scale(x_test)
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 4
epochs = 50

from keras.layers import BatchNormalization, LeakyReLU

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(512)(inputs)
x = BatchNormalization(momentum=0.8)(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(256)(x)
x = BatchNormalization(momentum=0.8)(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(128)(x)
x = BatchNormalization(momentum=0.8)(x)
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

from keras.initializers import TruncatedNormal

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(128)(latent_inputs)
x = BatchNormalization(momentum=0.8)(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(256)(x)
x = BatchNormalization(momentum=0.8)(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(512)(x)
x = BatchNormalization(momentum=0.8)(x)
x = LeakyReLU(alpha=0.2)(x)
outputs = Dense(original_dim, activation='tanh')(x)#,bias_initializer=TruncatedNormal(mean=-2,stddev=0.1))(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    help_ = "Load h5 model trained weights"
#    parser.add_argument("-w", "--weights", help=help_)
#    help_ = "Use mse loss instead of binary cross entropy (default)"
#    parser.add_argument("-m",
#                        "--mse",
#                        help=help_, action='store_true')
#    args = parser.parse_args()
models = (encoder, decoder)
data = x_train

    # VAE loss = mse_loss or xent_loss + kl_loss
#    if args.mse:
reconstruction_loss = mse(inputs, outputs)
#    else:
#reconstruction_loss = binary_crossentropy(inputs,
#                                                  outputs)
#
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)



vae.add_loss(vae_loss)

#import keras.optimizers.Adam

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=True)

vae.compile(optimizer=adam)#'adam')
vae.summary()
plot_model(vae,
           to_file='vae_mlp.png',
           show_shapes=True)


    
for j in range(20):
    print("macro_epoch_batch_"+str(j))
    vae.fit(x_train[0:1000,:],#[np.random.randint(low=0,high=x_train.shape[0],size=1000),:],
                epochs=1,
                batch_size=1)
    sample_images(models,
                 data,
                 batch_size=1,
                 epoch=j)
    
    
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=True)
vae.compile(optimizer=adam)
for j in range(21,40):
    print("macro_epoch_batch_"+str(j))
    vae.fit(x_train[0:1000,:],#[np.random.randint(low=0,high=x_train.shape[0],size=1000),:],
                epochs=1,
                batch_size=1)
    sample_images(models,
                 data,
                 batch_size=1,
                 epoch=j)
    
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=True)
vae.compile(optimizer=adam)
for j in range(1):
    print("macro_epoch_batch_"+str(j))
    vae.fit(x_train,#[np.random.randint(low=0,high=x_train.shape[0],size=1000),:],
                epochs=1,
                batch_size=1)
    sample_images(models,
                 data,
                 batch_size=1,
                 epoch=j)
    
adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, amsgrad=True)
vae.compile(optimizer=adam)
for j in range(1):
    print("macro_epoch_batch_"+str(j))
    vae.fit(x_train,#[np.random.randint(low=0,high=x_train.shape[0],size=1000),:],
                epochs=1,
                batch_size=1)
    sample_images(models,
                 data,
                 batch_size=1,
                 epoch=j)

adam = Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, amsgrad=True)
vae.compile(optimizer=adam)
for j in range(1):
    print("macro_epoch_batch_"+str(j))
    vae.fit(x_train,#[np.random.randint(low=0,high=x_train.shape[0],size=1000),:],
                epochs=1,
                batch_size=1)
    sample_images(models,
                 data,
                 batch_size=1,
                 epoch=j)

def load_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        
        with open(x_files[0],'rb') as x_file:
            x = pickle.load(x_file)
        
        for i in x_files[1:5]:
            print(i)
            with open(i,'rb') as x_file:
                print(i)
                xi = pickle.load(x_file)
                x = np.concatenate((x,xi),axis=0)
                print(x.shape)
        return(x)

x_train = load_data()
#x_test = load_data()

#image_size = x_train.shape[1]
original_dim = 17 * 24
x_train = np.reshape(x_train, [-1, original_dim])
#x_test = np.reshape(x_test, [-1, original_dim])
x_train = scale(x_train)
    
for j in range(1):
    print("macro_epoch_batch_"+str(j))
    vae.fit(x_train,#[np.random.randint(low=0,high=x_train.shape[0],size=1000),:],
                epochs=1,
                batch_size=1)
    sample_images(models,
                 data,
                 batch_size=1,
                 epoch=j)
    
#        ,
#                validation_data=(x_test, None))
#        vae.save_weights('vae_mlp_mnist.h5')

    sample_images(models,
                 data,
                 batch_size=1,
                 epoch=j)
#    sample_images2(models,j)
#    ,
#                 model_name="vae_mlp_epoch_"+str(j))
    
for j in range(20,200):
    print("macro_epoch_batch_"+str(j))
    vae.fit(x_train,#[np.random.randint(low=0,high=x_train.shape[0],size=1000),:],
                epochs=1,
                batch_size=128)
#        ,
#                validation_data=(x_test, None))
#        vae.save_weights('vae_mlp_mnist.h5')

    sample_images(models,
                 data,
                 batch_size=1,
                 epoch=j)

    
def sample_images2(models):
#        r, c = 5, 5
        noise = np.random.normal(0, 1, (2000000, 4))
        gen_imgs = decoder.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = (gen_imgs-np.min(gen_imgs))/(np.max(gen_imgs)-np.min(gen_imgs))
        gen_imgs = gen_imgs.reshape(-1,17,24,1)
        np.save("simulated_data/vae.npy",arr=gen_imgs)

sample_images2(models)



        

#def sample_images(models,data,epoch, latent_dim=2,batch_size=500000):
#    noise = np.random.normal(0, 1, (batch_size, latent_dim))
#    gen_imgs = decoder.predict(noise)
##    gen_imgs = 0.5 * gen_imgs + 0.5
#    gen_imgs = (gen_imgs-np.min(gen_imgs))/(np.max(gen_imgs)-np.min(gen_imgs))
#    gen_imgs = gen_imgs.reshape(-1,17,24,1)
#    # Rescale images 0 - 1
#    for i in range(batch_size):
#        
#        plt.imshow(gen_imgs[i,:,:,0],cmap='gray')
#        plt.savefig("images/vae"+str(epoch)+".png")
#        plt.close()
        
        