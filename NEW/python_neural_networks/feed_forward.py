# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:37:05 2019

@author: gerhard
"""

import glob

#list python dictionary files in specified directory

files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\unprocessed\\000265309" + '/**/*.txt', recursive=True)

#make an iterator variable to loop through files

a = list(range(1,len(files)-1))

#need to investigate whether this is still applicable
#a.append(0)

files_in_order = []
for i in a:
    files_in_order.append(files[i])
    
from ast import literal_eval

#initialize empty dictionary

d = {}

#open each file and append the missing closing bracket '}', perform literal_eval to read in the python dictionary contained in the file
#and append the dictionary just read in to the main dictionary

for i in range(0,len(files_in_order)):
            print(files_in_order[i])
            di = open(files_in_order[i])
            di = di.read()
            di = di + "}"
            di = literal_eval(di)
            d.update(di)
            
#get a list of the available variables in each dictionary
names = list(d.get(1).keys())

names.sort()

print(names)

#get the dictionary keys

k = d.keys()

#for each of the available variables, extract these into appropriate arrays for further processing

Eta = []

P = []

PT = []

Phi = []

Theta = []

dEdX = []

pdgCode = []

for i in k:
    Eta_i = d.get(i).get('Eta')
    Pi = d.get(i).get('P')
    pdg_i = d.get(i).get('pdgCode')
    
    P.append(Pi)
    pdgCode.append(pdg_i)
    
electron = []

for i in pdgCode:
    if abs(i)==11:
        electron.append(1)
    else:
        electron.append(0)
        
len(electron)
        
el_dist=[]
pi_dist=[]

for i in range(0,len(electron)):
    if electron[i]==1:
        el_dist.append(P[i])
    else:
        pi_dist.append(P[i])
        
    
print(P)

import matplotlib.pyplot as plt

colors = ['red','blue']

plt.hist([el_dist, pi_dist], density=False, bins=30, color = colors)