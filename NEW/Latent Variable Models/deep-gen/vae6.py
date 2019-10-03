print("==============================================================================================")

print("starting........................................................................................")

import glob

import numpy as np

print("imported glob, np........................................................................................")

#x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/x_*.pkl")
#y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/y_*.pkl")

x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\1_8_to_2_2_GeV\\x_*.pkl")

import pickle

print("loading first x pickle........................................................................................")

with open(x_files[0], 'rb') as x_file0:
    x = pickle.load(x_file0)
    x.shape = (x.shape[1],x.shape[2],x.shape[3])
   
print("recursively adding x pickles........................................................................................")

for i in x_files[1:]:
    with open(i,'rb') as x_file:
        print(i)
        xi = pickle.load(x_file)
        xi.shape = (xi.shape[1],xi.shape[2],xi.shape[3])
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

x=x.astype('float')

for i in range(0,x.shape[0]):
    ma = np.max(x[i,:,:])
    x[i,:,:]=x[i,:,:]/ma

x.shape = (x.shape[0],408)

#mu = np.mean(x)
#
#x = x.astype('float32')/mu

from sklearn.model_selection import train_test_split

x_train, x_test = train_test_split(x, test_size=0.2,random_state=123456)

import tensorflow as tf

tf.reset_default_graph()

batch_size = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 17, 24], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 17, 24], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 17 * 24])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 12

#reshaped_dim = [-1, 7, 7, dec_in_channels]
#inputs_decoder = 49 * dec_in_channels / 2

reshaped_dim = [-1, 8,8, dec_in_channels]
inputs_decoder = 64 * dec_in_channels / 2

def lrelu(x, alpha=0.3):
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
#        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
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
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

import matplotlib.pyplot as plt


start=0
stop=start+64
for i in range(100000):
    #batch = [np.reshape(b, [17, 24]) for b in x_train.next_batch(batch_size=batch_size)[0]]
    if stop>x_train.shape[0]:
        break
    batch=x[start:stop,:]
    batch=np.reshape(batch,[64,17,24])
    start=stop
    stop=start+64
    sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
        
    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
        plt.imshow(np.reshape(batch[0], [17, 24]), cmap='gray')
        plt.show()
        plt.savefig('C:\\Users\\gerhard\\Documents\\deep-gen/vae6_train'+str(i)+'_real'+'.png', bbox_inches='tight')
        plt.imshow(d[0], cmap='gray')
        plt.show()
        plt.savefig('C:\\Users\\gerhard\\Documents\\deep-gen/vae6_train'+str(i)+'_gen'+'.png', bbox_inches='tight')
        print(i, ls, np.mean(i_ls), np.mean(d_ls))



randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
imgs = [np.reshape(imgs[i], [17, 24]) for i in range(len(imgs))]

i=0

for img in imgs:
    i=i+1
    plt.figure(figsize=(17,24))
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.savefig('C:/Users/Gerhard/Documents/deep-gen/vae6_res'+str(i)+'.png', bbox_inches='tight')


















