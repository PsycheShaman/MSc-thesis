# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy
import tensorflow as tf


with tf.Session() as sess:
  devices = sess.list_devices()
  
print(devices)

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
    
 #   'C:\Users\gerhard\AppData\Roaming\Python\Python35\Scripts'
 
tf.test.is_gpu_available(cuda_only=True,min_cuda_compute_capability=None)