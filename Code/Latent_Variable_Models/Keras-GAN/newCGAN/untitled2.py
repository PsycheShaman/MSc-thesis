
from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt

import numpy as np

import pickle
import glob

import keras

def load_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        
        with open(x_files[0],'rb') as x_file:
            x = pickle.load(x_file)
        
        for i in x_files[1:]:
            print(i)
            with open(i,'rb') as x_file:
                print(i)
                xi = pickle.load(x_file)
                x = np.concatenate((x,xi),axis=0)
                print(x.shape)
        
        y_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\y_*.pkl")
        
        with open(y_files[0],'rb') as y_file:
            y = pickle.load(y_file)
        
        for i in y_files[1:]:
            print(i)
            with open(i,'rb') as y_file:
                print(i)
                yi = pickle.load(y_file)
                y = np.concatenate((y,yi),axis=0)
                print(y.shape)
        
        charge_sum = np.sum(x,axis=(1,2))
        
        for i in range(0,len(y)):
            if np.abs(y[i])==11:
                y[i] = 1
            else:
                y[i] = 0
        
        return(x,y,charge_sum)

(x,y ,charge_sum) = load_data()

del x
del y
        
def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def smooth_positive_labels(y):
	return y - 0.3 + (np.random.random(y.shape) * 0.5)

def smooth_negative_labels(y):
	return y + np.random.random(y.shape) * 0.3

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 17
        self.img_cols = 24
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 2
        self.latent_dim = 4

        optimizer_gen = Adam(0.000002, 0.5)
        optimizer_discr = SGD(0.000004)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer_discr,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        charge_sum = Input(shape=(1,))
        img = self.generator([noise, label,charge_sum])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label,charge_sum])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label,charge_sum], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer_gen)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh',bias_initializer=\
                        keras.initializers.TruncatedNormal(mean=-2, stddev=0.2, seed=None)))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        charge_sum = Input(shape=(1,),dtype='float32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
#        cs_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(charge_sum))

        model_input = multiply([noise, label_embedding,charge_sum])
        img = model(model_input)

        return Model([noise, label,charge_sum], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')
        charge_sum = Input(shape=(1,), dtype='float32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
#        cs_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(charge_sum))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding,charge_sum])

        validity = model(model_input)

        return Model([img, label,charge_sum], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train,charge_sum) = load_data()

        X_train = X_train.astype(np.float32)

        # Rescale -1 to 1
        X_train = scale(X_train)
#        charge_sum = scale(charge_sum)
        
        X_train = np.expand_dims(X_train, axis=3)
        
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        valid = smooth_positive_labels(valid)
        fake = np.zeros((batch_size, 1))
        fake = smooth_negative_labels(fake)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels,charge_sums = X_train[idx], y_train[idx], charge_sum[idx]/np.max(charge_sum)

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels,charge_sums])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels,charge_sums], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels,charge_sums], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
            sampled_charge_sums = np.random.choice(charge_sum,batch_size).reshape(-1, 1)
            sampled_charge_sums = sampled_charge_sums/np.max(charge_sum)
#            uniform(-1, 1, batch_size)
            
            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels,sampled_charge_sums], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            if epoch >= 1000000:
                if epoch % 1000000 == 0:
                    self.sample_images2(epoch)

    def sample_images(self, epoch):
            noise = np.random.normal(0, 1, (2, self.latent_dim))
            labels = np.array((0,1)).reshape(-1, 1)
            charge_sums = np.random.choice(charge_sum,2).reshape(-1, 1)
            gen_imgs = self.generator.predict([noise, labels,charge_sums/np.max(charge_sum)])
    
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            plt.imshow(gen_imgs[0,:,:,0],cmap='gray')
            plt.title(label="Particle: pi "+"; Charge sum: "+str(charge_sums[0])+"; After epoch: "+str(epoch))
            plt.savefig("C:/Users/gerhard/documents/keras-gan/newCGAN/images/pion_%d.png" % epoch)
            plt.close()
            plt.imshow(gen_imgs[1,:,:,0],cmap='gray')
            plt.title(label="Particle: e"+"; Charge sum: "+str(charge_sums[1])+"; After epoch: "+str(epoch))
            plt.savefig("C:/Users/gerhard/documents/keras-gan/newCGAN/images/electron_%d.png" % epoch)
            plt.close()
        
    def sample_images2(self, epoch):
#        r, c = 5, 5
        noise = np.random.normal(0, 1, (1000000, self.latent_dim))
        labels = np.array((np.ones(500000),np.zeros(500000))).reshape(-1, 1)
        
#        labels = np.random.randint(0, self.num_classes,1000000).reshape(-1, 1)
        charge_sums = np.random.choice(charge_sum,1000000).reshape(-1, 1)
        gen_imgs = self.generator.predict([noise, labels,charge_sums/np.max(charge_sum)])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        np.save("C:/Users/gerhard/documents/keras-gan/newCGAN/simulated_data/"+str(epoch)+".npy",arr=gen_imgs)


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=50000000000000000000001, batch_size=32, sample_interval=10)