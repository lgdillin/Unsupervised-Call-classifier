import pandas as pd
import numpy as np

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import preprocessing
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
# from sklearn.mixture import GMM
from sklearn.neighbors.kde import KernelDensity
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

plt.style.use('classic')

cellphone_athena = pd.read_csv('./data/test_transform.csv')
# landline_athena = pd.read_csv('./data/landline_athena_anomaly.csv')

#cellphone_athena = cellphone_athena.drop(['LineNumber', 'CallCategory'], axis=1)


#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
#cellphone_athena = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width', 'target'])
#cellphone_athena = cellphone_athena.drop(['target'], axis=1)

## Scale the data to have mean=0 and unit variance:
# scaler = RobustScaler().fit(cellphone_athena)
scaler = StandardScaler().fit(cellphone_athena)
#scaler = MinMaxScaler().fit(cellphone_athena)
scaler.transform(cellphone_athena)

#Plotting the Cumulative Summation of the Explained Variance
'''
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()
'''

# Apply the transformation
num_comp = 27
pca = PCA()
cellphone_transformed = pca.fit_transform(cellphone_athena)
#cellphone_athena['PCA1'] = cellphone_transformed[:, 0]
#cellphone_athena['PCA2'] = cellphone_transformed[:, 1]
print(pca.get_covariance())
print('==============================')
#print(pca.components_.shape)
# print(pca.components_)

#sns.lmplot('PCA1', 'PCA2', data=cellp
# hone_athena, fit_reg=False)
labels_list = range(0, num_comp)
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


#scaler1 = MinMaxScaler().fit(cellphone_transformed)
#scaler1.transform(cellphone_transformed)
#df = pd.DataFrame(cellphone_transformed)
#df.to_csv('./data/PCA_trans.csv', index=False)

#reduced = TSNE(n_components=1).fit_transform(cellphone_athena)
#cellphone_athena['TSNE1'] = reduced[:, 0]
#plt.plot(cellphone_athena['TSNE1'])
#plt.show()

