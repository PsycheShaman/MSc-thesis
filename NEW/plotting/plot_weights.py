# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:15:58 2019

@author: gerhard
"""


import tensorflow as tf

#def encoder(X_in, keep_prob):
#    activation = lrelu
#    with tf.variable_scope("encoder", reuse=None):
#        X = tf.reshape(X_in, shape=[-1, 17, 24, 1])
#        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
#        x = tf.nn.dropout(x, keep_prob)
#        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
#        x = tf.nn.dropout(x, keep_prob)
#        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
#        x = tf.nn.dropout(x, keep_prob)
#        x = tf.contrib.layers.flatten(x)
#        mn = tf.layers.dense(x, units=n_latent)
#        sd       = 0.5 * tf.layers.dense(x, units=n_latent)            
#        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
#        z  = mn + tf.multiply(epsilon, tf.exp(sd))
#        
#        return z, mn, sd

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=4,strides=2,input_shape=(17,24,1),padding='same'))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=4,strides=2,padding='same'))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=4,strides=2,padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100))

from keras.utils import plot_model
plot_model(model, to_file='C:/Users/gerhard/Documents/MSc-thesis/encoder.png',show_shapes=True,show_layer_names=False)

#def decoder(sampled_z, keep_prob):
#    with tf.variable_scope("decoder", reuse=None):
#        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
##        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
#        x = tf.layers.dense(x, units=inputs_decoder * 2, activation=lrelu)
#        x = tf.reshape(x, reshaped_dim)
#        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
#        x = tf.nn.dropout(x, keep_prob)
#        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
#        x = tf.nn.dropout(x, keep_prob)
#        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
#        
#        x = tf.contrib.layers.flatten(x)
#        x = tf.layers.dense(x, units=17*24, activation=tf.nn.sigmoid)
#        img = tf.reshape(x, shape=[-1, 17, 24])
#        return img


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32,input_shape=(100,)))
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Reshape(target_shape=(8,8,1)))
model.add(tf.keras.layers.Conv2DTranspose(64,kernel_size=4,strides=2,padding='same'))
model.add(tf.keras.layers.Conv2DTranspose(64,kernel_size=4,strides=2,padding='same'))
model.add(tf.keras.layers.Conv2DTranspose(64,kernel_size=4,strides=2,padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(17*24))
model.add(tf.keras.layers.Reshape(target_shape=(17,24)))

from keras.utils import plot_model
plot_model(model, to_file='C:/Users/gerhard/Documents/MSc-thesis/decoder.png',show_shapes=True,show_layer_names=False)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(2,3),activation="tanh",name="16filters8x2",input_shape=(17,24,1)))
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation="tanh"))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(128,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

from keras.utils import plot_model
plot_model(model, to_file='C:/Users/gerhard/Documents/MSc-thesis/gvr.png',show_shapes=True,show_layer_names=False)

model = tf.keras.models.load_model("C:/Users/gerhard/Documents/hpc-mini/final/model4_.h5")

weights = model.get_weights()

import numpy as np

import matplotlib.pyplot as plt

weights[0] = np.reshape(weights[0],(8,12,16))

plt.imshow(weights[0][:,:,1])

fig, axs = plt.subplots(4,4, figsize=(5,5))
fig.subplots_adjust(hspace = .5, wspace=.1)
axs = axs.ravel()
for i in range(16):
   axs[i].imshow(weights[0][:,:,i])
   axs[i].set_title(str(i))
   
  

weights[2] = np.reshape(weights[2],(5,6,16*32))

fig, axs = plt.subplots(32,16, figsize=(5,5))
fig.subplots_adjust(hspace = .2, wspace=.1)
axs = axs.ravel()
for i in range(512):
    axs[i].set_axis_off()
    axs[i].imshow(weights[2][:,:,i])
#   axs[i].set_title(str(i))

weights[4] = np.reshape(weights[4],(3,4,32*64))

fig, axs = plt.subplots(32,64, figsize=(100,100))
fig.subplots_adjust(hspace = .2, wspace=.1)
axs = axs.ravel()
for i in range(2048):
    axs[i].set_axis_off()
    axs[i].imshow(weights[4][:,:,i])

weights[6].shape

weights[6] = np.reshape(weights[6],(2,3,64*128))

fig, axs = plt.subplots(64,128, figsize=(100,100))
fig.subplots_adjust(hspace = .2, wspace=.1)
axs = axs.ravel()
for i in range(8192):
    axs[i].set_axis_off()
    axs[i].imshow(weights[6][:,:,i])


def plot_conv_weights(model, layer):
    W = model.layers[layer].get_weights()[0]
    if len(W.shape) == 4:
        W = np.squeeze(W)
        W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3])) 
        fig, axs = plt.subplots(8,4, figsize=(8,8))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for i in range(32):
            axs[i].imshow(W[:,:,i])
            axs[i].set_title(str(i))

plot_conv_weights(model,layer=1)


import pylab as pl
import matplotlib.cm as cm
import numpy.ma as ma
def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """

    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic
# utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)
#pl.imshow(make_mosaic(np.random.random((9, 10, 10)), 3, 3, border=1))

def plot_conv_weights(model, layer):
    # Visualize weights
    W = model.layers[layer].get_weights()
    W = np.squeeze(W)

    if len(W.shape) == 4:
        W = W.reshape((-1,W.shape[2],W.shape[3]))
    print("W shape : ", W.shape)

    pl.figure(figsize=(15, 15))
    pl.title('conv weights')
    s = int(np.sqrt(W.shape[0])+1)
    nice_imshow(pl.gca(), make_mosaic(W, s, s), cmap=cm.binary)

# usage
plot_conv_weights(model, layer=1)