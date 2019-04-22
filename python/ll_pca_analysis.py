import pandas as pd
import numpy as np

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import preprocessing
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, MDS
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

plt.style.use('classic')

cellphone_athena = pd.read_csv('./data/ll_transformed.csv')
# landline_athena = pd.read_csv('./data/landline_athena_anomaly.csv')
#cellphone_athena = cellphone_athena.drop(['LineNumber', 'CallCategory'], axis=1)

## Scale the data to have mean=0 and unit variance:
scaler = StandardScaler().fit(cellphone_athena)
scaler.transform(cellphone_athena)

print(cellphone_athena.shape[1])

# Apply the transformation
# num_comp = 12
pca = PCA()
cellphone_transformed = pca.fit_transform(cellphone_athena)
# Plot the variance elbow

# '''
print(np.cumsum(pca.explained_variance_ratio_))
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.xticks(np.arange(1, cellphone_athena.shape[1]+1, 1.0))
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()
# '''

# '''
labels_list = range(0, cellphone_athena.shape[1])
labels_names = []
for i in labels_list:
    labels_names.append('{}'.format(i))
plt.matshow(pca.get_covariance(),cmap='viridis')
plt.yticks(labels_list,cellphone_athena.columns.tolist(),fontsize=10)
plt.colorbar()
plt.xticks(range(len(cellphone_athena.columns.tolist())),
    cellphone_athena.columns.tolist(),rotation=65,ha='left')
plt.tight_layout()
plt.show()# 
# '''