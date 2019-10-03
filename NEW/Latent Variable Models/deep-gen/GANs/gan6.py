import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model,Sequential
from tqdm import tqdm
import glob
import pickle

def load_data():
    x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
    y_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\y_*.pkl")
    
 
    with open(x_files[0], 'rb') as x_file0:
        x = pickle.load(x_file0)

    with open(y_files[0], 'rb') as y_file0:
       y = pickle.load(y_file0)
 
    for i in x_files[1:]:
        with open(i,'rb') as x_file:
            print(i)
            xi = pickle.load(x_file)
            x = np.concatenate((x,xi),axis=0)
       
    for i in y_files[1:]:
        with open(i,'rb') as y_file:
            yi = pickle.load(y_file)
            y = np.concatenate((y,yi),axis=None)
            
    x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.npy")
    y_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\y_*.npy")
 
    for i in x_files[0:]:
        with open(i,'rb') as x_file:
            print(i)
            xi = np.load(x_file)
            x = np.concatenate((x,xi),axis=0)

    for i in y_files[0:]:
        with open(i,'rb') as y_file:
            yi = np.load(y_file)
            y = np.concatenate((y,yi),axis=None)

    x = x.reshape((-1, 17*24))

    
    return (x,y)

(x,y) = load_data()

print(x.shape)
print(y.shape)


def adam_optimizer():
    return tf.keras.optimizers.Adam(lr=0.0001)

def create_generator():
    generator=Sequential()
    generator.add(tf.keras.layers.Reshape((10,10,1), input_shape=(100,)))
    generator.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=1, padding='valid',activation="relu"))
    generator.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=1, padding='valid',activation="relu"))
    generator.add(tf.keras.layers.Flatten())
    
    generator.add(Dense(units=300,activation='relu'))
    
    generator.add(Dense(units=350,activation='relu'))
    
    generator.add(Dense(units=400,activation='relu'))
    
    generator.add(Dense(units=450,activation='relu'))
    
    generator.add(Dense(units=500,activation='relu'))
    
    generator.add(Dense(units=600,activation='relu'))
    
    generator.add(Dense(units=800,activation='relu'))
    
    generator.add(Dense(units=1000,activation='relu'))
    
    generator.add(Dense(units=408, activation='tanh'))
    
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator

g=create_generator()
g.summary()

def create_discriminator():
    discriminator=Sequential()
    discriminator.add(tf.keras.layers.Reshape((17,24,1), input_shape=(408,)))
    discriminator.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(8,12),strides=1, padding='valid',activation="relu"))
    discriminator.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(5,6),strides=1, padding='valid',activation="relu"))
    discriminator.add(tf.keras.layers.Flatten())
    discriminator.add(tf.keras.layers.Dense(256,activation="relu"))
    discriminator.add(tf.keras.layers.Dense(128,activation="relu"))
    discriminator.add(tf.keras.layers.Dense(64,activation="relu"))
    discriminator.add(tf.keras.layers.Dense(32,activation="relu"))
    discriminator.add(tf.keras.layers.Dense(6,activation="sigmoid"))
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator

d =create_discriminator()
d.summary()

def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
#    x = tf.reshape(x,[-1,17,24,1])
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

gan = create_gan(d,g)
gan.summary()

def plot_generated_images(epoch, generator, examples=1, dim=(10,10), figsize=(17,24)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(1,17,24)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        #plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('gan6_generated_image %d.png' %epoch)


def training(epochs=1, batch_size=365):
    
    #Loading the data
    (x,y) = load_data()
    batch_count = x.shape[0] / batch_size
    
    # Creating GAN
    generator= create_generator()
    discriminator= create_discriminator()
    gan = create_gan(discriminator, generator)
    
    for e in range(1,epochs+1 ):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
        #generate  random noise as an input  to  initialize the  generator
            noise= np.random.normal(0,1, [batch_size, 100])
            
            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)
            
            # Get a random set of  real images
            image_batch =x[np.random.randint(low=0,high=x.shape[0],size=batch_size)]
            
            #Construct different batches of  real and fake data 
            X= np.concatenate([image_batch, generated_images])
            
            # Labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9
            
            #Pre train discriminator on  fake and real data  before starting the gan. 
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            
            #Tricking the noised input of the Generator as real data
            noise= np.random.normal(0,1, [batch_size, 100])
            y_gen = np.ones(batch_size)
            
            # During the training of gan, 
            # the weights of discriminator should be fixed. 
            #We can enforce that by setting the trainable flag
            discriminator.trainable=False
            
            #training  the GAN by alternating the training of the Discriminator 
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)
            
        if e == 1 or e % 1 == 0:
           
            plot_generated_images(e, generator)
training(500,10000)










