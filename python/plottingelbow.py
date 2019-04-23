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

pcavar = [0.3801, 0.5941, 0.7180, 0.79496, 0.84751, 0.89259, 0.92615, 0.95244, 0.97519, 0.9896, 0.99684, 1.00000]
plt.figure()
plt.plot(pcavar)
plt.xlabel('Number of Components')
plt.xticks(np.arange(1, 11, 1.0))
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()
# '''