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

nSigmaElectron = []

nSigmaPion = []


layer0 = []

layer1 = []

layer2 = []

layer3 = []

layer4 = []

layer5 = []

for i in k:
    Eta_i = d.get(i).get('Eta')
    P_i = d.get(i).get('P')
    PT_i = d.get(i).get('PT')
    Phi_i = d.get(i).get('Phi')
    Theta_i = d.get(i).get('Theta')
    dEdX_i = d.get(i).get('dEdX')
    pdgCode_i = d.get(i).get('pdgCode')
    nSigmaElectron_i = d.get(i).get('nSigmaElectron')
    nSigmaPion_i = d.get(i).get('nSigmaPion')
    
    layer0_i = d.get(i).get('layer 0')
    layer1_i = d.get(i).get('layer 1')
    layer2_i = d.get(i).get('layer 2')
    layer3_i = d.get(i).get('layer 3')
    layer4_i = d.get(i).get('layer 4')
    layer5_i = d.get(i).get('layer 5')
    
    Eta.append(Eta_i)
    P.append(P_i)
    PT.append(PT_i)
    Phi.append(Phi_i)
    Theta.append(Theta_i)
    dEdX.append(dEdX_i)
    pdgCode.append(pdgCode_i)
    nSigmaElectron.append(nSigmaElectron_i)
    nSigmaPion.append(nSigmaPion_i)
    
    layer0.append(layer0_i)
    layer1.append(layer1_i)
    layer2.append(layer2_i)
    layer3.append(layer3_i)
    layer4.append(layer4_i)
    layer5.append(layer5_i)
    
    
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
        

import matplotlib.pyplot as plt

colors = ['red','blue']

plt.xlabel('Momentum (GeV)')
plt.ylabel('Entries')
plt.title('Histogram of Electron/ Pion Momentum')

plt.hist([el_dist, pi_dist], density=False, bins=30, color = colors, label = ['electron','pion'])

import numpy as np

layer0[1]
layer0[2]
layer0[3]
layer0[4]
layer0[5]
layer0[6]
layer0[7]
layer0[8]
layer0[9]
type(layer0[10])
type(layer0[11])
layer0[12]
layer0[13]
layer0[14]

layer0[15]

a =[]

a.append(np.average(layer0[15],axis=1).tolist())
a.append(np.average(layer1[15],axis=1).tolist())



j=15
newlist = []

L = len(np.average(layer0[j],axis=1).tolist())

signal = 0

for i in range(L):
    if i == 0 and np.average(layer0[j],axis=1).tolist()[i]<=10:
        newlist.append(0)
    elif i == 0:
        if signal == 0:
            signal = signal+1
            newlist.append(signal)
    else:
        if np.average(layer0[j],axis=1).tolist()[i]<=11:
            newlist.append(0)
        elif np.average(layer0[j],axis=1).tolist()[i]>11:
            if signal == 0:
                signal = signal + 1
                newlist.append(signal)
            elif newlist[i-1]==signal:
                newlist.append(signal)
            else:
                signal = signal +1
                newlist.append(signal)


a[0]

len(a)


layer1[15]
layer2[15]
layer3[13]
layer4[13]
layer5[13]

layer0[16]
layer0[17]
layer0[18]
layer0[19]
layer0[20]
layer0[21]











