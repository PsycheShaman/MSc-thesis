# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:59:27 2019

@author: Gerhard
"""

print("==============================================================================================")

print("starting........................................................................................")

import glob

import numpy as np

print("imported glob, np........................................................................................")

x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/x_*.pkl")
y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/y_*.pkl")

#x_files = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/ff/x_*.pkl")
#y_files = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/ff/y_*.pkl")

import pickle

print("loading first x pickle........................................................................................")

with open(x_files[0], 'rb') as x_file0:
    x = pickle.load(x_file0)
    
print("loading first y pickle........................................................................................")

with open(y_files[0], 'rb') as y_file0:
   y = pickle.load(y_file0)
   
print("recursively adding x pickles........................................................................................")

for i in x_files[1:]:
#for i in x_files[1:2]:
    with open(i,'rb') as x_file:
        xi = pickle.load(x_file)
        x = np.concatenate((x,xi),axis=0)
        
print("recursively adding y pickles........................................................................................")
        
for i in y_files[1:]:
#for i in y_files[1:2]:
    with open(i,'rb') as y_file:
        yi = pickle.load(y_file)
        y = np.concatenate((y,yi),axis=None)
        
nz = np.array([np.count_nonzero(i) for i in x])

zeros = np.where(nz==0)

x = np.delete(x,zeros,axis=0)
y = np.delete(y,zeros)

#oversample electrons

elec = np.where(y==1)
pion = np.where(y!=1)

electrons_x = x[elec,:,:]

electrons_y = y[elec]

electrons_x = np.squeeze(electrons_x)

x = np.concatenate((electrons_x,x,electrons_x),axis=0)

y = np.concatenate((electrons_y,y,electrons_y),axis=None)

        
print(x.shape)

print(y.shape)
        
print("dumping x pickles........................................................................................")
        
#with open('C:/Users/gerhard/Documents/x.pkl', 'wb') as x_file:
#  pickle.dump(x, x_file)

with open('/scratch/vljchr004/data/msc-thesis-data/ff/x.pkl', 'wb') as x_file:
  pickle.dump(x, x_file)
  
print("dumping y pickles........................................................................................")

#with open('C:/Users/gerhard/Documents/y.pkl', 'wb') as y_file:
#  pickle.dump(y, y_file)

with open('/scratch/vljchr004/data/msc-thesis-data/ff/y.pkl', 'wb') as y_file:
  pickle.dump(y, y_file)
  
print("**************************************<--done-->***************************************************")
  
  
  
  
  
  
  
  
  
  
