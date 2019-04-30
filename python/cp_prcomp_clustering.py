import pandas as pd
import numpy as np
import umap

import matplotlib as mpl 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, MDS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

plt.style.use('classic')

print('Loading Data')
cellphone_prcomp = pd.read_csv('./data/cp_pca.csv')
cellphone_subset = cellphone_prcomp[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']]

print('Running TSNE')
reduced = TSNE(n_components=2, 
                random_state=0, 
                n_iter=3500,
                early_exaggeration=41.0,
                perplexity=71,
                learning_rate=150
                ).fit_transform(cellphone_subset)

print('Clustering Data')
cluster = KMeans(n_clusters = 3, random_state=2).fit(cellphone_subset)
labels = cluster.predict(cellphone_subset)

print('Generating Plot')
cellphone_prcomp['DIM1'] = reduced[:, 0]
cellphone_prcomp['DIM2'] = reduced[:, 1]
cellphone_prcomp['labels'] = labels

sns.lmplot('DIM1', 'DIM2', data=cellphone_prcomp, hue = 'labels', fit_reg=False, scatter_kws={"s": 100})
plt.show()
