# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:23:36 2019

@author: gerhard
"""

import glob

files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\unprocessed\\000265309" + '/**/*.txt', recursive=True)

a = list(range(1,len(files)-1))

#a.append(0)

files_in_order = []
for i in a:
    files_in_order.append(files[i])
    
from ast import literal_eval

d = {}

for i in range(0,len(files_in_order)):
            print(files_in_order[i])
            di = open(files_in_order[i])
            di = di.read()
            di = di + "}"
            di = literal_eval(di)
            d.update(di)
       
names = list(d.get(1).keys())

names.sort()

print(names)

k = d.keys()

momentum = []

pdg = []

for i in k:
    P = d.get(i).get('P')
    pdg_i = d.get(i).get('pdgCode')
    
    momentum.append(P)
    pdg.append(pdg_i)
    
electron = []

for i in pdg:
    if abs(i)==11:
        electron.append(1)
    else:
        electron.append(0)
        
len(electron)
        
el_dist=[]
pi_dist=[]

for i in range(0,len(electron)):
    if electron[i]==1:
        el_dist.append(momentum[i])
    else:
        pi_dist.append(momentum[i])
        
    
print(momentum)

import matplotlib.pyplot as plt

colors = ['red','blue']

plt.hist([el_dist, pi_dist], density=False, bins=30, color = colors)

































