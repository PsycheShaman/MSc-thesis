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

#el = np.where(labels==1)
#pi = np.where(labels==0)
#
#el_sample = np.random.randint(low=0,high=len(el[0]),size=50000)
#pi_sample = np.random.randint(low=0,high=len(pi[0]),size=50000)
#
#el = el[0]
#pi = pi[0]
#
#el = el[el_sample]
#pi = pi[pi_sample]
#
#x_el = tracks[el,:,:,:]
#x_pi = tracks[pi,:,:,:]
#
#x = np.vstack((x_el,x_pi))
#
#del tracks
#
#y_el = labels[el]
#y_pi = labels[pi]
#
#
#y = np.concatenate((y_el,y_pi))

x= tracks

del tracks

y= labels

del labels

from sklearn.manifold import TSNE
import seaborn as sns

x.shape = (x.shape[0],17*24)

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

from mpl_toolkits.mplot3d import Axes3D

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"], 
    ys=df.loc[rndperm,:]["pca-two"], 
    zs=df.loc[rndperm,:]["pca-three"], 
    c=df.loc[rndperm,:]["y"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

#Kernel PCA

ind = np.concatenate((np.array(range(0,40)),np.array(range(100,140))))

x2 = x[ind,:]
y2 = y[ind]

from sklearn.decomposition import KernelPCA

scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1)
X_skernpca = scikit_kpca.fit_transform(x2)

plt.figure(figsize=(8,6))
plt.scatter(X_skernpca[y2==0, 0], X_skernpca[y2==0, 1], color='red', alpha=0.8)
plt.scatter(X_skernpca[y2==1, 0], X_skernpca[y2==1, 1], color='blue', alpha=0.8)

plt.text(-0.48, 0.35, 'gamma = 15', fontsize=12)
plt.title('First 2 principal components after RBF Kernel PCA via scikit-learn')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#def stepwise_kpca(X, gamma, n_components):
#    """
#    Implementation of a RBF kernel PCA.
#
#    Arguments:
#        X: A MxN dataset as NumPy array where the samples are stored as rows (M),
#           and the attributes defined as columns (N).
#        gamma: A free parameter (coefficient) for the RBF kernel.
#        n_components: The number of components to be returned.
#
#    """
#    # Calculating the squared Euclidean distances for every pair of points
#    # in the MxN dimensional dataset.
#    sq_dists = pdist(X, 'sqeuclidean')
#
#    # Converting the pairwise distances into a symmetric MxM matrix.
#    mat_sq_dists = squareform(sq_dists)
#
#    # Computing the MxM kernel matrix.
#    K = exp(-gamma * mat_sq_dists)
#
#    # Centering the symmetric NxN kernel matrix.
#    N = K.shape[0]
#    one_n = np.ones((N,N)) / N
#    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
#
#    # Obtaining eigenvalues in descending order with corresponding
#    # eigenvectors from the symmetric matrix.
#    eigvals, eigvecs = eigh(K)
#
#    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
#    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
#
#    return X_pc
#
#X_pc = stepwise_kpca(x2, gamma=15, n_components=1)
#
#plt.figure(figsize=(8,6))
#plt.scatter(X_pc[y==0, 0], np.zeros((500,1)), color='red', alpha=0.5)
#plt.scatter(X_pc[y==1, 0], np.zeros((500,1)), color='blue', alpha=0.5)
#plt.text(-0.05, 0.007, 'gamma = 15', fontsize=12)
#plt.title('First principal component after RBF Kernel PCA')
#plt.xlabel('PC1')
#plt.show()




#T-SNE

N = x.shape[0]
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values


pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.8
)
































