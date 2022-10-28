
from pyexpat import model
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, plot_confusion_matrix,f1_score,precision_score,accuracy_score
#machine algorithms
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

#from sklearn.svm import SVR
import utils
import math
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn import neighbors
from sklearn import metrics

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

scaring={'accuracy','balanced_accuracy','roc_auc','f1','neg_mean_absolute_error','neg_root_mean_squared_error','r2'}

param_C_linear = 10000
param_C_poly = 100
param_C_rbf = 1000
param_C_sigmoid = 1000 


def f_importances(coef, names):
    imp = coef[0]
    imp,names = zip(*sorted(zip(imp,names)))
    # plt.barh(range(len(names)), imp, align='center')
    # plt.yticks(range(len(names)), names)
    # plt.show()

def normalize_data(data_x, data_y):
    scalerx = MinMaxScaler().fit(data_x)
    X_std = scalerx.fit_transform(data_x)
    scalery = MinMaxScaler() 
    Y_std = data_y.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X_std, Y_std, 
                                            train_size = 0.80,random_state=0)

    return X_train, X_test, y_train, y_test

def aznormalize_data(data_x, data_y):
    scalerx = MinMaxScaler().fit(data_x)
    X_std = scalerx.fit_transform(data_x)
    scalery = MinMaxScaler() 
    Y_std = data_y.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X_std, Y_std, 
                                            train_size = 0.80,random_state=0)

    return X_train, X_test, y_train, y_test,X_std,Y_std


def azmodeluygula(data_x, data_y, feature_names, k, plotting):

    

    models=[]
    models.append(('SVML',svm.SVC(gamma=0.001,  C=param_C_linear, kernel = 'linear')))
    models.append(('SVMP',svm.SVC(C=param_C_poly, degree=3, kernel="poly")))
    models.append(('SVMRBF',svm.SVC(C=param_C_rbf, kernel="rbf")))
    models.append(('SVMSig',svm.SVC(C=param_C_sigmoid, kernel="sigmoid")))
    models.append(('DT',DecisionTreeClassifier()))
    models.append(('RF',RandomForestClassifier(n_estimators=100,criterion='gini')))
    #models.append(('NB',GaussianNB()))
    #models.append(('KNN',neighbors.KNeighborsClassifier(5, weights="distance", algorithm="kd_tree")))

    #print(models)
    X_train, X_test, y_train, y_test,X_std,Y_std = aznormalize_data(data_x, data_y)
    Y_std=Y_std.ravel()
    results=dict()
    names=[]
    table1=[]
    tsutun = ['Model', 'Değişken_sayisi','cv_ort','trainscore','testscore','f1','precision','recall','rocauc']
    tabledict=dict()
    for name,model in models:
        kfold=KFold(n_splits=10,random_state=7,shuffle=True)
        cvresults_acc=cross_val_score(model,X_std,Y_std,cv=10,scoring="accuracy")
        cvresults_f1=cross_val_score(model,X_std,Y_std,cv=10,scoring="f1_macro")
        cvresults_preci=cross_val_score(model,X_std,Y_std,cv=10,scoring="precision")
        cvresults_recall=cross_val_score(model,X_std,Y_std,cv=10,scoring="recall")
        cvresults_rocauc=cross_val_score(model,X_std,Y_std,cv=10,scoring="roc_auc")
        results[name]=(cvresults_acc.mean(),cvresults_acc.std(),cvresults_f1.mean(),cvresults_preci.mean(),cvresults_recall.mean(),cvresults_rocauc.mean())
        names.append(name)

        model=model.fit(X_train,y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        table1.append([name,k,cvresults_acc.mean(),train_score,test_score,cvresults_f1.mean(),cvresults_preci.mean(),cvresults_recall.mean(),cvresults_rocauc.mean()])
        tabledict[name]=(name,k,cvresults_acc.mean(),train_score,test_score)
    print()
    print("name   results.mean      result.std ")

    #for key,value in results.items():
    #    print(key,value)

    for sonuc in table1:
        print(sonuc)
    for key,value in tabledict.items():
        print(key,value)
    
    return test_score, train_score,table1,tabledict

def azgridcv(data_x, data_y, feature_names, k, plotting):
    model_params={
        'SVM':{
            'model':svm.SVC(gamma='auto'),
            'params':{
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }
        },
        'RF':{
            'model':RandomForestClassifier(),
            'params':{
                'n_estimators':[1,5,10]
            }
        }

    }
    #model paramları bitti

    models=[]
    models.append(('SVML',svm.SVC(gamma=0.001,  C=param_C_linear, kernel = 'linear')))
    models.append(('SVMP',svm.SVC(C=param_C_poly, degree=3, kernel="poly")))
    models.append(('SVMRBF',svm.SVC(C=param_C_rbf, kernel="rbf")))
    models.append(('SVMSig',svm.SVC(C=param_C_sigmoid, kernel="sigmoid")))
    models.append(('DT',DecisionTreeClassifier()))
    models.append(('RF',RandomForestClassifier(n_estimators=100,criterion='gini')))
    #models.append(('NB',GaussianNB()))
    #models.append(('KNN',neighbors.KNeighborsClassifier(5, weights="distance", algorithm="kd_tree")))

    #print(models)
    X_train, X_test, y_train, y_test,X_std,Y_std = aznormalize_data(data_x, data_y)
    Y_std=Y_std.ravel()
    results=dict()
    names=[]
    table1=[]
    tsutun = ['Model', 'Değişken_sayisi','cv_ort','trainscore','testscore','f1','precision','recall','rocauc']
    tabledict=dict()
    for name,model in models:
        kfold=KFold(n_splits=10,random_state=7,shuffle=True)
        cvresults_acc=cross_val_score(model,X_std,Y_std,cv=10,scoring="accuracy")
        cvresults_f1=cross_val_score(model,X_std,Y_std,cv=10,scoring="f1_macro")
        cvresults_preci=cross_val_score(model,X_std,Y_std,cv=10,scoring="precision")
        cvresults_recall=cross_val_score(model,X_std,Y_std,cv=10,scoring="recall")
        cvresults_rocauc=cross_val_score(model,X_std,Y_std,cv=10,scoring="roc_auc")
        results[name]=(cvresults_acc.mean(),cvresults_acc.std(),cvresults_f1.mean(),cvresults_preci.mean(),cvresults_recall.mean(),cvresults_rocauc.mean())
        names.append(name)

        model=model.fit(X_train,y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        table1.append([name,k,cvresults_acc.mean(),train_score,test_score,cvresults_f1.mean(),cvresults_preci.mean(),cvresults_recall.mean(),cvresults_rocauc.mean()])
        tabledict[name]=(name,k,cvresults_acc.mean(),train_score,test_score)
    print()
    print("name   results.mean      result.std ")

    #for key,value in results.items():
    #    print(key,value)

    for sonuc in table1:
        print(sonuc)
    for key,value in tabledict.items():
        print(key,value)
        
    return test_score, train_score,table1,tabledict