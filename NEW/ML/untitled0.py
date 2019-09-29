# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:34:14 2019

@author: gerhard
"""

import numpy as np

x = np.load("C:/Users/Gerhard/Documents/msc-thesis-data/simulated_datasets/simulated_data_aae12/sim.npy")

x.shape = (1000000,17*24)

np.savetxt(fname="C:/Users/Gerhard/Documents/msc-thesis-data/simulated_datasets/simulated_data_aae12/sim_flat.csv",X=x,delimiter=","\
           ,fmt="%1.10f",newline="\n")