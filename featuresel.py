import pandas as pd
import utils

from sklearn.datasets import load_breast_cancer
#cancer= load_breast_cancer()
#can=pd.DataFrame(cancer.data, columns=cancer.feature_names)
data_x, data_y, columnfeaturesx, data_ysinif = utils.loadfirma("erdemveri01")
X=data_x
y=data_y
feature_names=columnfeaturesx
#X, y = load_breast_cancer(return_X_y=True)
import mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.linear_model import LogisticRegression
# Sequential Forward Selection(sfs)
sfs = SFS(LogisticRegression(),
           k_features=22,
           forward=True,
           floating=False,
           scoring = 'accuracy',
           cv = 0)
sfs1=sfs.fit(X, y)
fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')
result_LR = pd.DataFrame.from_dict(sfs1.get_metric_dict(confidence_interval=0.90)).T
result_LR.sort_values('avg_score', ascending=0, inplace=True)
result_LR.head()

best_features_LR = result_LR.feature_idx.head(1).tolist()
select_features_LR = data_x.columns[best_features_LR]
select_features_LR
print(select_features_LR)