import glob

import numpy as np

import pickle

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
        return(x)
        
x = load_data()

x = x.astype('float32')

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

x = scale(x)

x.shape = (x.shape[0],408)

import tensorflow as tf

tf.reset_default_graph()

batch_size = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 17, 24], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 17, 24], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 17 * 24])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 4

reshaped_dim = [-1, 8,8, dec_in_channels]
inputs_decoder = 64 * dec_in_channels / 2

def lrelu(x, alpha=0.2):
    return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 17, 24, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd       = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        
        return z, mn, sd

def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=17*24, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 17, 24])
        return img
    
sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

unreshaped = tf.reshape(dec, [-1, 17*24])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.00001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

import matplotlib.pyplot as plt


start=0
stop=start+64
for i in range(10000000):
    #batch = [np.reshape(b, [17, 24]) for b in x_train.next_batch(batch_size=batch_size)[0]]
    if stop>x.shape[0]:
        start=0
        stop=start+64
    batch=x[start:stop,:]
    batch=np.reshape(batch,[64,17,24])
    start=stop
    stop=start+64
    sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
        
    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
        
        plt.savefig('C:\\Users\\gerhard\\Documents\\Keras-GAN/newVAE/images/'+str(i)+'_gen'+'.png', bbox_inches='tight')
        plt.show()
        print(i, ls, np.mean(i_ls), np.mean(d_ls))


for r in range(0,1000):
    randoms = [np.random.normal(0, 1, n_latent) for _ in range(1000)]
    imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
    imgs = [np.reshape(imgs[i], [17, 24]) for i in range(len(imgs))]
    
    np.save(file="C:/Users/Gerhard/Documents/Keras-GAN/newVAE/simulated_data/"+str(r)+".npy",arr=imgs)
    
#    del imgs

i=0

for img in imgs:
    i=i+1
    plt.figure(figsize=(17,24))
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.savefig('C:/Users/Gerhard/Documents/Keras-GAN/newVAE/images/'+str(i)+'.png', bbox_inches='tight')


















