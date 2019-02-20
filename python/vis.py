import pandas as pd

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

plt.style.use('classic')

athena = pd.read_csv('./data/UA_AthenaData_1.csv')

athena = athena.drop(['LineNumber', 'CallCategory'], axis=1)
athena.head()

# model = PCA(n_components=2)
model = Isomap(n_components=2)
model.fit(athena)
athena_2d = model.transform(athena)

athena['PCA1'] = athena_2d[:, 0]
athena['PCA2'] = athena_2d[:, 1]

sns.lmplot('PCA1', 'PCA2', data=athena, fit_reg=False)
plt.show()