# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 19:52:36 2019

@author: gerhard
"""

import numpy as np

tracks = np.load("C:/Users/gerhard/documents/6_tracklets_large_calib_train/0_tracks.npy")

infosets = np.load("C:/Users/gerhard/documents/6_tracklets_large_calib_train/0_info_set.npy")

tracks = tracks.reshape((-1, 17, 24, 1))


labels = np.repeat(infosets[:, 0], 6)

el = np.where(labels==1)
pi = np.where(labels==0)

el_sample = np.random.randint(low=0,high=len(el[0]),size=50000)
pi_sample = np.random.randint(low=0,high=len(pi[0]),size=50000)

el = el[0]
pi = pi[0]

el = el[el_sample]
pi = pi[pi_sample]

x_el = tracks[el,:,:,:]
x_pi = tracks[pi,:,:,:]

x = np.vstack((x_el,x_pi))

del tracks

y_el = labels[el]
y_pi = labels[pi]


y = np.concatenate((y_el,y_pi))

from sklearn.manifold import TSNE
import seaborn as sns

x.shape = (100000,17*24)

import pandas as pd

feat_cols = [ 'pixel'+str(i) for i in range(x.shape[1]) ]
df = pd.DataFrame(x,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))

import matplotlib.pyplot as plt

rndperm = np.random.permutation(df.shape[0])

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.8
)














