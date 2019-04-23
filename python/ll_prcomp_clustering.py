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

# load cellphone data
landline_prcomp = pd.read_csv('./data/ll_pca.csv')
landline_subset = landline_prcomp[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]

# Data is already scaled from R's prcomp()
# scaler = StandardScaler().fit(cellphone_athena)
# scaler.transform(cellphone_athena)


'''
sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k, n_init = 50, random_state = 0)
    km = km.fit(landline_prcomp)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k (landline)')
plt.show()
'''

reduced = umap.UMAP(n_components = 3, n_neighbors=20, min_dist=0.15).fit_transform(landline_subset)

#cluster = GaussianMixture(n_components = 2).fit(landline_subset)
cluster = KMeans(n_clusters = 3, precompute_distances = True, algorithm = 'full').fit(landline_subset)
# cluster = AgglomerativeClustering(n_clusters = 2, linkage = 'single')
labels = cluster.predict(landline_subset)
# labels = DBSCAN(eps = 1.0, min_samples = 10).fit_predict(landline_subset)


landline_prcomp['DIM1'] = reduced[:, 0]
landline_prcomp['DIM2'] = reduced[:, 1]
landline_prcomp['DIM3'] = reduced[:, 2]
landline_prcomp['labels'] = labels

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(landline_prcomp['DIM1'], landline_prcomp['DIM2'], landline_prcomp['DIM3'])
# sns.lmplot('DIM1', 'DIM2', data=landline_prcomp, hue='labels',fit_reg=False)
plt.show()
