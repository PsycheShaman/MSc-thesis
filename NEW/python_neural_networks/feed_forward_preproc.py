# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:37:05 2019

@author: gerhard
"""

import glob

#list python dictionary files in specified directory

files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\unprocessed\\" + '\\**\\*.txt', recursive=True)

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

print("concatenating files.............................................................")

for i in range(0,len(files_in_order)):
            print(files_in_order[i])
            di = open(files_in_order[i])
            di = di.read()
            if di == "}":
                #print(di)
                continue
            else:
                di = di + "}"
                di = literal_eval(di)
                #print("length of di: "+str(len(di)))
                ki = list(di.keys())
                j = len(d)
                for k in ki:
                    j += 1000000000000000000000000000000123
                    print(str(k)+" becomes: "+str(j))
                    di[j] = di.pop(k)
                    #print("length of di after pop: "+str(len(di)))
                
                #print(len(di))
                    d.update(di)
            print(str(100*(i/len(files_in_order))))
            
#get a list of the available variables in each dictionary
#names = list(d.get(1).keys())
#
#names.sort()
#
#print(names)

#get the dictionary keys

k = d.keys()

#for each of the available variables, extract these into appropriate arrays for further processing

#Eta = []
#
#P = []
#
#PT = []
#
#Phi = []
#
#Theta = []

#dEdX = []

pdgCode = []

#nSigmaElectron = []
#
#nSigmaPion = []


layer0 = []

layer1 = []

layer2 = []

layer3 = []

layer4 = []

layer5 = []

print("extracting data from dictionaries................................................")

for i in k:
#    Eta_i = d.get(i).get('Eta')
#    P_i = d.get(i).get('P')
#    PT_i = d.get(i).get('PT')
#    Phi_i = d.get(i).get('Phi')
#    Theta_i = d.get(i).get('Theta')
#    dEdX_i = d.get(i).get('dEdX')
    pdgCode_i = d.get(i).get('pdgCode')
#    nSigmaElectron_i = d.get(i).get('nSigmaElectron')
#    nSigmaPion_i = d.get(i).get('nSigmaPion')
    
    layer0_i = d.get(i).get('layer 0')
    layer1_i = d.get(i).get('layer 1')
    layer2_i = d.get(i).get('layer 2')
    layer3_i = d.get(i).get('layer 3')
    layer4_i = d.get(i).get('layer 4')
    layer5_i = d.get(i).get('layer 5')
    
#    Eta.append(Eta_i)
#    P.append(P_i)
#    PT.append(PT_i)
#    Phi.append(Phi_i)
#    Theta.append(Theta_i)
#    dEdX.append(dEdX_i)
    pdgCode.append(pdgCode_i)
#    nSigmaElectron.append(nSigmaElectron_i)
#    nSigmaPion.append(nSigmaPion_i)
    
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
        
#len(electron)
#        
#el_dist=[]
#pi_dist=[]
#
#for i in range(0,len(electron)):
#    if electron[i]==1:
#        el_dist.append(P[i])
#    else:
#        pi_dist.append(P[i])
#        
#
#import matplotlib.pyplot as plt
#
#colors = ['red','blue']
#
#plt.xlabel('Momentum (GeV)')
#plt.ylabel('Entries')
#plt.title('Histogram of Electron/ Pion Momentum')
#
#plt.hist([el_dist, pi_dist], density=False, bins=30, color = colors, label = ['electron','pion'])

import numpy as np



#def signal_check(layer):
#    
#    zero_signal = []
#    ok_signal = []
#    dodgy_signal = []
#    
#    for j in range(len(layer)):
#        
#        if type(layer[j])==type(None) or np.array(layer[j]).shape==(17,0):
#            zero_signal.append(j)
#            
#        else:
#            newlist = []
#            
#            L = len(np.average(layer[j],axis=1).tolist())
#            
#            signal = 0
#            
#            for i in range(L):
#                if i == 0 and np.average(layer[j],axis=1).tolist()[i]==0:
#                    newlist.append(0)
#                elif i == 0:
#                    if signal == 0:
#                        signal = signal+1
#                        newlist.append(signal)
#                else:
#                    if np.average(layer[j],axis=1).tolist()[i]==0:
#                        newlist.append(0)
#                    elif np.average(layer[j],axis=1).tolist()[i]>0:
#                        if signal == 0:
#                            signal = signal + 1
#                            newlist.append(signal)
#                        elif newlist[i-1]==signal:
#                            newlist.append(signal)
#                        else:
#                            signal = signal +1
#                            newlist.append(signal)
#            
#            if max(newlist)==0:
#                zero_signal.append(j)
#            
#            elif max(newlist)==1:
#                ok_signal.append(j)
#            
#            else:
#                dodgy_signal.append(j)
#            
#    a = (ok_signal,dodgy_signal,zero_signal)
#            
#    return(a)
#
##check signal for layer 0 
#           
#layer0_signal = signal_check(layer=layer0)
#
#layer0_ok_signal = layer0_signal[0]
#layer0_dodgy_signal = layer0_signal[1]
#layer0_zero_signal = layer0_signal[2]
#
##check signal for layer 1
#
#layer1_signal = signal_check(layer=layer1)
#
#layer1_ok_signal = layer1_signal[0]
#layer1_dodgy_signal = layer1_signal[1]
#layer1_zero_signal = layer1_signal[2]
#
##check signal for layer 2
#
#layer2_signal = signal_check(layer=layer2)
#
#layer2_ok_signal = layer2_signal[0]
#layer2_dodgy_signal = layer2_signal[1]
#layer2_zero_signal = layer2_signal[2]
#
##check signal for layer 3
#
#layer3_signal = signal_check(layer=layer3)
#
#layer3_ok_signal = layer3_signal[0]
#layer3_dodgy_signal = layer3_signal[1]
#layer3_zero_signal = layer3_signal[2]
#
#
##check signal for layer 4
#
#layer4_signal = signal_check(layer=layer4)
#
#layer4_ok_signal = layer4_signal[0]
#layer4_dodgy_signal = layer4_signal[1]
#layer4_zero_signal = layer4_signal[2]
#
##check signal for layer 5
#
#layer5_signal = signal_check(layer=layer5)
#
#layer5_ok_signal = layer5_signal[0]
#layer5_dodgy_signal = layer5_signal[1]
#layer5_zero_signal = layer5_signal[2]

#len(layer0_ok_signal)+len(layer1_ok_signal)+len(layer2_ok_signal)+len(layer3_ok_signal)+len(layer4_ok_signal)+len(layer5_ok_signal)
    

#this is just to get the first non-null element to initialize the x and y arrays

print("getting x and y values................................................")

for i in range(len(layer0)):
    if type(layer0[i])==type(None) or np.array(layer0[i]).shape==(17,0):
        continue
    else:
        x = np.array(layer0[i])
        x = np.sum(x,axis=0)
        y = np.array(electron[i])
        beg=i
        break
    
for i in range(beg+1,len(layer0)):
    if type(layer0[i])==type(None) or np.array(layer0[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer0[i])
        xi = np.sum(xi,axis=0)
        yi = electron[i]
        x = np.concatenate((x,xi))
        y = np.append(y,yi)
    if type(layer1[i])==type(None) or np.array(layer1[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer1[i])
        xi = np.sum(xi,axis=0)
        #yi = electron[i]
        x = np.concatenate((x,xi))
        #y = np.append(y,yi)
    if type(layer2[i])==type(None) or np.array(layer2[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer2[i])
        xi = np.sum(xi,axis=0)
        #yi = electron[i]
        x = np.concatenate((x,xi))
        #y = np.append(y,yi)
    if type(layer3[i])==type(None) or np.array(layer3[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer3[i])
        xi = np.sum(xi,axis=0)
        #yi = electron[i]
        x = np.concatenate((x,xi))
        #y = np.append(y,yi)
    if type(layer4[i])==type(None) or np.array(layer4[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer4[i])
        xi = np.sum(xi,axis=0)
        #yi = electron[i]
        x = np.concatenate((x,xi))
        #y = np.append(y,yi)
    if type(layer5[i])==type(None) or np.array(layer5[i]).shape==(17,0):
        continue
    else:
        xi = np.array(layer5[i])
        xi = np.sum(xi,axis=0)
        #yi = electron[i]
        x = np.concatenate((x,xi))
        #y = np.append(y,yi)
    print(str(100*i/len(layer0)))
        
x = np.array(x)
y = np.array(y)

x = np.reshape(x,(len(y),24))
x = x.astype('float32')

mu = np.mean(x)
x /= mu

# Step 1
import pickle
 
with open('C:\\Users\\gerhard\\Documents\\msc-thesis-data\\x.full', 'wb') as x_file:
  pickle.dump(x, x_file)
  
with open('C:\\Users\\gerhard\\Documents\\msc-thesis-data\\y.full', 'wb') as y_file:
  pickle.dump(y, y_file)

























