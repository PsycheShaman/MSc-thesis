# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 07:33:20 2019

@author: gerhard
"""

def load_data():
    tracks = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_tracks.npy")

    infosets = np.load("C:/Users/Gerhard/Documents/6_tracklets_large_calib_train/0_info_set.npy")

    x = tracks.reshape((-1, 17*24))

    y = np.repeat(infosets[:, 0], 6)
    return (x,y)

(x,y) = load_data()

import numpy as np

np.savetxt(fname="C:/Users/Gerhard/Documents/x.csv",X=x)
np.savetxt(fname="C:/Users/Gerhard/Documents/y.csv",X=y)










