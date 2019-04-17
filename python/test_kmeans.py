import pandas as pd
import numpy as np
# import umap

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, MDS
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

plt.style.use('classic')

# load cellphone data
cellphone_athena = pd.read_csv('./data/test_transform1.csv')

# Scale our data
scaler = StandardScaler().fit(cellphone_athena)
scaler.transform(cellphone_athena)

# Apply the transformation
num_comp = 4
pca = PCA(n_components=num_comp)
cellphone_transformed = pca.fit_transform(cellphone_athena)

sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(cellphone_transformed)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()