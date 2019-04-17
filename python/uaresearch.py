import pandas as pd
import numpy as np
import umap

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import preprocessing
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, MDS
from sklearn.mixture import GMM, BayesianGaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

plt.style.use('classic')

cellphone_athena = pd.read_csv('./data/test_transform1.csv')
# landline_athena = pd.read_csv('./data/landline_athena_anomaly.csv')
#cellphone_athena = cellphone_athena.drop(['LineNumber', 'CallCategory'], axis=1)

## Scale the data to have mean=0 and unit variance:
# scaler = RobustScaler().fit(cellphone_athena)
scaler = StandardScaler().fit(cellphone_athena)
#scaler = MinMaxScaler().fit(cellphone_athena)
scaler.transform(cellphone_athena)



# Apply the transformation
num_comp = 12
pca = PCA(n_components=4)
cellphone_transformed = pca.fit_transform(cellphone_athena)

# cluster = KMeans(n_clusters = 2).fit(cellphone_transformed)
# cluster = GMM(n_components = 2).fit(cellphone_transformed)
cluster = BayesianGaussianMixture(
    n_components = 2, 
    covariance_type = 'tied',
    # weight_concentration_prior_type = 'dirichlet_process',
    ).fit(cellphone_transformed)
labels = cluster.predict(cellphone_transformed)


# Plot the variance elbow
'''
print(np.cumsum(pca.explained_variance_ratio_))
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.xticks(np.arange(1, num_comp+1, 1.0))
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()
'''

tsne = TSNE(n_components=2, random_state=1, perplexity=90, n_iter= 4000, learning_rate=800)
# reduced = tsne.fit_transform(cellphone_transformed)
# reduced = MDS(n_components=2).fit_transform(cellphone_transformed)
reduced = umap.UMAP(n_neighbors=20, min_dist=0.15).fit_transform(cellphone_transformed)

cellphone_athena['DIM1'] = reduced[:, 0]
cellphone_athena['DIM2'] = reduced[:, 1]
cellphone_athena['labels'] = labels
sns.lmplot('DIM1', 'DIM2', data=cellphone_athena, hue='labels',fit_reg=False)
plt.show()

'''
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
'''
