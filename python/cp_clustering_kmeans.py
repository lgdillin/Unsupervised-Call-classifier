import pandas as pd
import numpy as np
# import umap

import matplotlib as mpl 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, MDS
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

plt.style.use('classic')

# load cellphone data
cellphone_athena = pd.read_csv('./data/cp_transformed.csv')

# Scale our data
scaler = StandardScaler().fit(cellphone_athena)
scaler.transform(cellphone_athena)

# Apply the transformation
num_comp = 2
pca = PCA()
# print(pca.components_)
cellphone_transformed = pca.fit_transform(cellphone_athena)
# cellphone_transformed = pd.DataFrame(cellphone_transformed)
# print(cellphone_transformed[0])
print(pca.components_)
print(pca.components_[0][10])
print(pca.components_[1][4])
print(cellphone_athena.columns)

# '''
sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(cellphone_transformed)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k (cellphone)')
# plt.show()
# '''

# reduced = umap.UMAP(n_neighbors=20, min_dist=0.15).fit_transform(cellphone_transformed)

cluster = KMeans(n_clusters = 4).fit(cellphone_transformed)
labels = cluster.predict(cellphone_transformed)


cellphone_athena['DIM1'] = cellphone_transformed[:, 0]
cellphone_athena['DIM2'] = cellphone_transformed[:, 1]
# cellphone_athena['DIM3'] = cellphone_transformed[:, 2]
cellphone_athena['labels'] = labels

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
# ax.scatter(cellphone_athena['DIM1'], cellphone_athena['DIM2'], cellphone_athena['DIM3'])

sns.lmplot('DIM1', 'DIM2', data=cellphone_athena, hue='labels',fit_reg=False)
# sns.lmplot('Total_N30Day_Calls', 'Total_NToday_AvgDuration', data=cellphone_athena,fit_reg=False)
plt.show()
