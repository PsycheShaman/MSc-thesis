# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:36:16 2019

@author: gerhard
"""

import tensorflow as tf

from tensorflow import keras

import os

import matplotlib.pyplot as plt

def plot_results(models,
                 data,
                 batch_size=32,
                 model_name="vae_pretrained_1"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_train, y_train = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_train,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_train)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "C:/Users/gerhard/Documents/MSc-thesis/NEW/ML/vae_res/particles_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = [17,24]
    figure = np.zeros((digit_size[0] * n, digit_size[1] * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size[0], digit_size[1])
            figure[i * digit_size[0]: (i + 1) * digit_size[0],
                   j * digit_size[1]: (j + 1) * digit_size[1]] = digit

    plt.figure(figsize=(10, 10))
    start_range = 17*24 // 2
    end_range = (n - 1) * 17*24 + start_range + 1
    pixel_range = np.arange(start_range, end_range, 17*24)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


"""
model <- keras_model_sequential() %>%
  layer_conv_2d(input_shape = c(17,24,1),filters = 32,kernel_size = c(6,6),padding = "same",activation = "tanh") %>%
  layer_max_pooling_2d() %>%
  layer_dropout(rate=0.2) %>%
  layer_conv_2d(64,c(3,3),activation = "tanh",padding="same") %>%
  layer_max_pooling_2d() %>%
  layer_dropout(rate=0.2) %>%
  layer_conv_2d(128,c(3,3),activation = "tanh",padding="same") %>%
  layer_max_pooling_2d() %>%
  layer_dropout(rate=0.2) %>%
  layer_flatten() %>%
  layer_dense(512,"tanh") %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(256,"tanh") %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(128,"tanh") %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(1,activation = "sigmoid")
"""


model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(input_shape=(17,24,1),filters=32,kernel_size=(6,6),padding="same",activation="tanh"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="tanh"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",activation="tanh"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,"tanh"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(256,"tanh"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(128,"tanh"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(1,"sigmoid"))

model.load_weights("C:/Users/gerhard/Documents/MSc-thesis/NEW/ML/model_focal_loss_lte_2_GeV.h5")

weights  = model.get_weights()

inputs = tf.keras.Input(shape=(17,24,1))

conved_inputs = tf.keras.layers.Conv2D(filters=32,kernel_size=(6,6),\
                                       padding="same",activation="tanh",weights=(weights[0],weights[1]))(inputs)
conved_inputs = tf.keras.layers.MaxPool2D()(conved_inputs)
conved_inputs = tf.keras.layers.Dropout(rate=0.2)(conved_inputs)


conved_inputs2 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="tanh",\
                                        weights=(weights[2],weights[3]))(conved_inputs)
conved_inputs2 = tf.keras.layers.MaxPool2D()(conved_inputs2)
conved_inputs2 = tf.keras.layers.Dropout(rate=0.2)(conved_inputs2)

conved_inputs3 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",activation="tanh",\
                                        weights=(weights[4],weights[5]))(conved_inputs2)
conved_inputs3 = tf.keras.layers.MaxPool2D()(conved_inputs3)
conved_inputs3 = tf.keras.layers.Dropout(rate=0.2)(conved_inputs3)

flat = tf.keras.layers.Flatten()(conved_inputs3)

dense1 = tf.keras.layers.Dense(512,"tanh",weights=(weights[6],weights[7]))(flat)
dense1 = tf.keras.layers.Dropout(rate=0.2)(dense1)

dense2 = tf.keras.layers.Dense(256,"tanh",weights=(weights[8],weights[9]))(dense1)
dense2 = tf.keras.layers.Dropout(rate=0.2)(dense2)

dense3 = tf.keras.layers.Dense(128,"tanh")(dense2)
dense3 = tf.keras.layers.Dropout(rate=0.2)(dense3)

z_mean = tf.keras.layers.Dense(3, name='z_mean')(dense3)
z_log_var = tf.keras.layers.Dense(3, name='z_log_var')(dense3)

from keras import backend as K

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

z = tf.keras.layers.Lambda(sampling, output_shape=(3,), name='z')([z_mean, z_log_var])


encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()


tf.keras.utils.plot_model(encoder, show_shapes=True,to_file="C:/Users/gerhard/Documents/MSc-thesis/NEW/ML/pretrained_VAE_encoder.png")


# build decoder model
latent_inputs = tf.keras.layers.Input(shape=(3,), name='z_sampling')

dense_decoder = tf.keras.layers.Dense(128,"tanh")(latent_inputs)
dense_decoder = tf.keras.layers.Dropout(rate=0.2)(dense_decoder)

dense_decoder2 = tf.keras.layers.Dense(256,"tanh")(dense_decoder)
dense_decoder2 = tf.keras.layers.Dropout(rate=0.2)(dense_decoder2)

dense_decoder3 = tf.keras.layers.Dense(400,"tanh")(dense_decoder2)
dense_decoder3 = tf.keras.layers.Dropout(rate=0.2)(dense_decoder3)

reshaped = tf.keras.layers.Reshape(target_shape=(20,20,1))(dense_decoder3)
reshaped_conved = tf.keras.layers.Conv2DTranspose(128,(3,3),activation="tanh")(reshaped)

flattened_reshape = tf.keras.layers.Flatten()(reshaped_conved)
flattened_reshape = tf.keras.layers.Dense(512,"tanh")(flattened_reshape)
flattened_reshape = tf.keras.layers.Dropout(rate=0.2)(flattened_reshape)
outputs = tf.keras.layers.Dense(408)(flattened_reshape)
outputs = tf.keras.layers.Reshape(target_shape=(17,24,1))(outputs)

decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
decoder.summary()


tf.keras.utils.plot_model(decoder, to_file='C:/Users/gerhard/Documents/MSc-thesis/NEW/ML/pretrained_VAE_encoder.png', show_shapes=True)

outputs = decoder(encoder(inputs)[2])
vae = tf.keras.Model(inputs, outputs, name='vae')

reconstruction_loss = tf.keras.losses.mse(inputs, outputs)

reconstruction_loss *= 17*24
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
tf.keras.utils.plot_model(vae,
           to_file='C:/Users/gerhard/Documents/MSc-thesis/NEW/ML/pretrained_VAE_full.png',
           show_shapes=True)

import numpy as np

def load_data():
        tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")
    
        infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")
    
        x = tracks.reshape((-1, 17,24,1))
        
        
#        x = tracks
    
        y = np.repeat(infosets[:, 0], 6)
        return (x,y)
    
(x_train,y_train) = load_data()

vae.fit(x_train,
                epochs=6000000,
                batch_size=1)

vae.save_weights('C:/Users/gerhard/Documents/MSc-thesis/NEW/ML/vae_pretrained_1.h5')



#plot_results(models,
#                 data,
#                 batch_size=32)



















