import pandas as pd
import numpy as np

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns

#import umap 
from sklearn import preprocessing
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
#from sklearn.mixture import GMM
from sklearn.preprocessing import RobustScaler

# load all the data
df0 = pd.read_csv('./data/labeled_data_0.csv')
# df1 = pd.read_csv('./data/labeled_data_1.csv')
# df2 = pd.read_csv('./data/labeled_data_2.csv')
# df3 = pd.read_csv('./data/labeled_data_3.csv')
# Place into list of dataframes
# frames = [df0, df1, df2, df3]
print('data loaded')

cluster = DBSCAN(eps=3, min_samples=2).fit(df0[['PCA1', 'PCA2']])
print('Generating plot')
df0 = df0.drop(columns=['cluster'])
df0['cluster'] = cluster.labels_
sns.lmplot('PCA1', 'PCA2', data=df0, hue='cluster', fit_reg=False)
plt.show()


'''
for i in range(len(frames)):
    print('creating temp frame')
    print('clustering')
    cluster = DBSCAN(eps=3, min_samples=2).fit(frames[i][['PCA1', 'PCA2']])
    print('Generating plot')
    frames[i] = frames[i].drop(columns=['cluster'])
    frames[i]['cluster'] = cluster.labels_
    sns.lmplot('PCA1', 'PCA2', data=frames[i], hue='cluster', fit_reg=False)
    plt.show()
'''
# Print the output
#sns.lmplot('PCA1', 'PCA2', data=subdata, hue='cluster', fit_reg=False)
#plt.show()