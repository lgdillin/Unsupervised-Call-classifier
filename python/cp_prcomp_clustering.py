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
cellphone_prcomp = pd.read_csv('./data/cp_pca.csv')
cellphone_subset = cellphone_prcomp[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]#, 'PC6']]
print(cellphone_subset.columns)
# Data is already scaled from R's prcomp()
# scaler = StandardScaler().fit(cellphone_athena)
# scaler.transform(cellphone_athena)


#'''
sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k, n_init = 50, random_state = 0)
    km = km.fit(cellphone_subset)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k (cellphone)')
plt.show()
#'''

# reduced = umap.UMAP(n_neighbors=20, min_dist=0.1).fit_transform(cellphone_subset)
reduced = umap.UMAP().fit_transform(cellphone_subset)

# cluster = GaussianMixture(n_components = 2).fit(cellphone_subset)
cluster = KMeans(n_clusters = 3).fit(cellphone_subset)
# cluster = AgglomerativeClustering(n_clusters = 2, linkage = 'single').fit(cellphone_subset)
labels = DBSCAN().fit_predict(cellphone_subset)
# labels = cluster.predict(cellphone_subset)


cellphone_prcomp['DIM1'] = reduced[:, 0]
cellphone_prcomp['DIM2'] = reduced[:, 1]
cellphone_prcomp['labels'] = labels

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(cellphone_athena['DIM1'], cellphone_athena['DIM2'], cellphone_athena['DIM3'])
sns.lmplot('DIM1', 'DIM2', data=cellphone_prcomp, hue = 'labels', fit_reg=False)
plt.show()
