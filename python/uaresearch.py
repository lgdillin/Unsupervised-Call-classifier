import pandas as pd
import numpy as np

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns

import umap 
from sklearn import preprocessing
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.mixture import GMM
from sklearn.preprocessing import RobustScaler, StandardScaler

plt.style.use('classic')

cellphone_athena = pd.read_csv('./data/cellphone_athena_anomaly.csv')
# landline_athena = pd.read_csv('./data/landline_athena_anomaly.csv')

cellphone_athena = cellphone_athena.drop(['LineNumber', 'CallCategory'], axis=1)

## Scale the data to have mean=0 and unit variance:
# scaler = RobustScaler().fit(cellphone_athena)
scaler = StandardScaler().fit(cellphone_athena)
scaler.transform(cellphone_athena)

pca = PCA(n_components=0.85)
pca.fit(cellphone_athena)

print(pca.components_)
print(pca.explained_variance_)

