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
from sklearn.preprocessing import RobustScaler

plt.style.use('classic')

# Load the CSV data
athena = pd.read_csv('./data/UA_AthenaData.csv')

# Drop the encrypted phone number (LineNumber), and the Call category (As labeled by data team)
athena = athena.drop(['LineNumber', 'CallCategory'], axis=1)

# Split into subgroups, as training on the entire dataset breaks my computer
group = np.array_split(athena, 4)


# Iterate through each group
for i in range(len(group)):
    print('======= GROUP {} ======'.format(i))
    subdata = group[i]

    ## Scale the data to have mean=0 and unit variance:
    print('Scaling Data')
    scaler = RobustScaler().fit(athena)
    scaler.transform(athena)

    ## Reduce data for clustering
    print('Reducing dimensions')
    model = umap.UMAP(n_neighbors=20, min_dist=0.15, metric='braycurtis')
    data_2d = model.fit_transform(subdata)

    print('Clustering Data')
    cluster = DBSCAN(eps=3, min_samples=2).fit(subdata)

    print('Configuring data to clusters')
    subdata['PCA1'] = data_2d[:, 0]
    subdata['PCA2'] = data_2d[:, 1]
    cluster.labels_[cluster.labels_ > 0] = 1
    subdata['cluster'] = cluster.labels_

    print('Saving to file')
    group[i].to_csv('./data/labeled_data_{}.csv'.format(i), index=False)


for i in range(len(group)):
    print('======= Drawing plot for group {} ======'.format(i))
    sns.lmplot('PCA1', 'PCA2', data=group[i], hue='cluster', fit_reg=False)
    plt.show()


'''
# athena = pd.read_csv('./data/UA_AthenaData_2.csv')
# athena = pd.read_csv('./data/UA_AthenaData_1.csv')
# athena = pd.read_csv('./data/output_test.csv')


## Scale the data to have mean=0 and unit variance:
#scaler = RobustScaler().fit(athena)
#scaler.transform(athena)


## Different models
#model = PCA(n_components=2)
#model = TSNE(n_components=2)
model = umap.UMAP(n_neighbors=20, min_dist=0.15, metric='braycurtis')

#model.fit(athena)
athena_2d = model.fit_transform(athena)

## Cluster
# cluster = AgglomerativeClustering().fit(athena_2d)
# cluster = AgglomerativeClustering(n_clusters=2, linkage='complete').fit(athena)
# cluster = GMM(n_components=16, covariance_type='full', random_state=0).fit_predict(athena)
# cluster = SpectralClustering(n_clusters=2, random_state=0 ).fit(athena)
cluster = DBSCAN(eps=3, min_samples=2).fit(athena)

athena['PCA1'] = athena_2d[:, 0]
athena['PCA2'] = athena_2d[:, 1]
#athena.to_csv('./data/output_test_1.csv', columns=['PCA1', 'PCA2'], index=False)
cluster.labels_[cluster.labels_ > 0] = 1
athena['cluster'] = cluster.labels_


sns.lmplot('PCA1', 'PCA2', data=athena, hue='cluster', fit_reg=False)
plt.show()

athena.to_csv('./data/labeled_1.csv', index=False)
'''