# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 16:03:21 2019

@author: gerhard
"""

import tensorflow.keras 

import numpy as np




tracks = np.load("C:/Users/gerhard/Documents/msc-thesis-data/jeremy/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("C:/Users/gerhard/Documents/msc-thesis-data/jeremy/6_tracklets_large_calib_train/0_info_set.npy")

train = tracks.reshape((-1, 17, 24, 1))

labels = np.repeat(infosets[:, 0], 6)

ma = np.max(train)

train = train/ma

model1 = tensorflow.keras.models.load_model('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model25_.h5')

model1.probs = model1.predict_proba(train)

model2 = tensorflow.keras.models.load_model('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model25_1.h5')

model2.probs = model2.predict_proba(train)

model3 = tensorflow.keras.models.load_model('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model25_2.h5')

model3.probs = model3.predict_proba(train)

model4 = tensorflow.keras.models.load_model('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model25_1_1.h5')

model4.probs = model4.predict_proba(train)

model5 = tensorflow.keras.models.load_model('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model25_1_2.h5')

model5.probs = model5.predict_proba(train)

model6 = tensorflow.keras.models.load_model('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model25_1_3.h5')

model6.probs = model6.predict_proba(train)

model7 = tensorflow.keras.models.load_model('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model25_1_4.h5')

model7.probs = model7.predict_proba(train)

model8 = tensorflow.keras.models.load_model('C:/Users/gerhard/Documents/hpc-mini/chamber_gain_corrected/model25_1_5.h5')

model8.probs = model8.predict_proba(train)

test = model1.probs*model2.probs*model3.probs*model4.probs*model5.probs*model6.probs*model7.probs*model8.probs

np.max(test)
np.min(test)

test2 = (model1.probs+model2.probs+model3.probs+model4.probs+model5.probs+model6.probs+model7.probs+model8.probs)/8

import matplotlib.pyplot as plt

plt.hist(test)

plt.hist(test2)

np.savetxt("c:/Users/gerhard/Documents/MSc-thesis/NEW/ML/ensemble1_out.csv",test)

np.savetxt("c:/Users/gerhard/Documents/MSc-thesis/NEW/ML/ensemble2_out.csv",test2)












