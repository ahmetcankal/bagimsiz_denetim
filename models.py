
from pyexpat import model
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import svm
#from sklearn.svm import SVR
import utils
import math
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2, f_regression



def f_importances(coef, names):
    imp = coef[0]
    imp,names = zip(*sorted(zip(imp,names)))
    # plt.barh(range(len(names)), imp, align='center')
    # plt.yticks(range(len(names)), names)
    # plt.show()



def main(k):
    # Select gov
    # "ireland","italy","greece","portugal","spain"
    data_x, data_y, feature_names = utils.loaddata(k)
    #model = LogisticRegression(solver='liblinear', random_state=0).fit(data_x,data_y)
  
   


    #################### SVC sınıflandırma işlemi#######
    # X = data_x
    # Y = data_ysinif
    # features_names = feature_names

    ############ sınıflandırma bölümü bitti  veriler normalize ediliyor ##########
  

    scalerx = MinMaxScaler().fit(data_x)
    X_std = scalerx.fit_transform(data_x)
    scalery = MinMaxScaler()
   # Y_std = scalery.fit_transform(data_y.reshape(-1,1))
    Y_std = data_y.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X_std, Y_std, train_size = 0.80,random_state=0)

    #X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, train_size = 0.70,random_state=0)

   



    #model yüzde 70 doğru sınıfladı
    svmmodel = svm.SVR(gamma=0.001, C=100, kernel = 'linear') 
    svmmodel.fit(X_train, y_train)
    # #pd.Series(abs(svm.coef_[0]), index=features.columns).nlargest(10).plot(kind='barh')
    y_pred = svmmodel.predict(X_test)
    print(svmmodel.score(X_test, y_test))

    testScore = svmmodel.score(X_test,y_test)
    print("test score:")
    print(testScore)

    Trainscore = svmmodel.score(X_train,y_train)
    print("train score:")
    print(Trainscore)

    # logreg = LogisticRegression(C=1e5)
    # logreg.fit(X_train, y_train)

    # y_predl=logreg.predict(X_test)
    y_pred = svmmodel.predict(X_test)
    ####################################
   # Plot the hyperplane
    # cml = confusion_matrix(y_test, y_predl)
    # print(cml)
    # print(logreg.score(X_test, y_test))

    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # print(svmmodel.score(X_test, y_test))
    f_importances(svmmodel.coef_, feature_names)

    # report = classification_report(y_test, y_pred)
    # print(report)

    return testScore,Trainscore