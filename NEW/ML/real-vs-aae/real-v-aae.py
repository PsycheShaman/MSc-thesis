# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:39:15 2019

@author: gerhard
"""

import glob

import numpy as np

import pickle

def load_real_data():
        x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\cnn\\x_*.pkl")
        
        with open(x_files[0],'rb') as x_file:
            x = pickle.load(x_file)
        return(x)


def load_simulated_data():
    x_files = glob.glob("C:\\Users\\gerhard\\Documents\\Keras-GAN\\newAAE\\v11\\simulated_data\\*.npy")
    
    with open(x_files[0],'rb') as x_file:
            x = np.load(x_file)
    
    for i in x_files[1:]:
        with open(i,'rb') as x_file:
            xi = np.load(x_file)
            x = np.concatenate((x,xi),axis=0)
        return(x)

sim = load_simulated_data()
real = load_real_data()





