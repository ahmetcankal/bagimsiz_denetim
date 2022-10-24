
from pyexpat import model
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, plot_confusion_matrix
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

############ nu support vector Machines ########################
def azsupport_vector_machine_linear(data_x, data_y, feature_names, k, plotting):

    X_train, X_test, y_train, y_test,X_std,Y_std = aznormalize_data(data_x, data_y)
    svmmodel = svm.SVC(gamma=0.001,  C=param_C_linear, kernel = 'linear') 
    svmodel=svmmodel.fit(X_train, y_train.ravel())
   
    train_score = svmodel.score(X_train, y_train)
    test_score = svmodel.score(X_test, y_test)


    lin_reg_model = svm.SVC( C=param_C_poly, degree=3, kernel="poly")
    lin_reg_model.fit(X_train, y_train.ravel())
    train_score = lin_reg_model.score(X_train, y_train)
    test_score = lin_reg_model.score(X_test, y_test)

    y_pred=lin_reg_model.predict(X_test)
    confusion_matrix(y_test,y_pred)
    plot_confusion_matrix(lin_reg_model,X_test,y_test,cmap=plt.cm.Blues)
    plt.show()
    print(classification_report(y_test,y_pred))


    if plotting:
        print("- train score:\t"+str(train_score)) 
        print("- test score:\t"+str(test_score))  

    return test_score, train_score



def azsupport_vector_machine_poly(data_x, data_y, feature_names, k, plotting):

    X_train, X_test, y_train, y_test,X_std,Y_std = aznormalize_data(data_x, data_y)

    lin_reg_model = svm.SVC( C=param_C_poly, degree=3, kernel="poly")
    lin_reg_model.fit(X_train, y_train.ravel())
    train_score = lin_reg_model.score(X_train, y_train)
    test_score = lin_reg_model.score(X_test, y_test)

    if plotting:
        print("- train score:\t"+str(train_score)) 
        print("- test score:\t"+str(test_score))  

    return test_score, train_score

def azsupport_vector_machine_rbf(data_x, data_y, feature_names, k, plotting):

    X_train, X_test, y_train, y_test,X_std,Y_std = aznormalize_data(data_x, data_y)

    lin_reg_model = svm.SVC( C=param_C_rbf, kernel="rbf")
    lin_reg_model.fit(X_train, y_train.ravel())
    train_score = lin_reg_model.score(X_train, y_train)
    test_score = lin_reg_model.score(X_test, y_test)

    if plotting:
        print("- train score:\t"+str(train_score)) 
        print("- test score:\t"+str(test_score))  

    return test_score, train_score

def azsupport_vector_machine_sigmoid(data_x, data_y, feature_names, k, plotting):

    X_train, X_test, y_train, y_test,X_std,Y_std = aznormalize_data(data_x, data_y)

    lin_reg_model = svm.SVC(C=param_C_sigmoid, kernel="sigmoid")
    lin_reg_model.fit(X_train, y_train.ravel())
    train_score = lin_reg_model.score(X_train, y_train)
    test_score = lin_reg_model.score(X_test, y_test)

    if plotting:
        print("- train score:\t"+str(train_score)) 
        print("- test score:\t"+str(test_score))  

    return test_score, train_score






    ################# Decision Tree class ##############################

def azDecision_Tree_class(data_x, data_y, feature_names, k, plotting):

    X_train, X_test, y_train, y_test,X_std,Y_std = aznormalize_data(data_x, data_y)

    
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train.ravel())
    train_score = dt_model.score(X_train, y_train)
    test_score = dt_model.score(X_test, y_test)

    if plotting:
        print("- train score:\t"+str(train_score)) 
        print("- test score:\t"+str(test_score))  

    return test_score, train_score

        ################# random forest class ##############################

def azRandom_forest(data_x, data_y, feature_names, k, plotting):

    X_train, X_test, y_train, y_test,X_std,Y_std = aznormalize_data(data_x, data_y)

    
    dt_model = RandomForestClassifier(n_estimators=100,criterion='gini')
    dt_model.fit(X_train, y_train.ravel())
    train_score = dt_model.score(X_train, y_train)
    test_score = dt_model.score(X_test, y_test)

    if plotting:
        print("- train score:\t"+str(train_score)) 
        print("- test score:\t"+str(test_score))  

    return test_score, train_score




    ### buraya kadar kullanıldı

            ################# naive bayes class ##############################

def naivebayes(data_x, data_y, feature_names, k, plotting):

    X_train, X_test, y_train, y_test = normalize_data(data_x, data_y)

    
    dt_model = GaussianNB()
    dt_model.fit(X_train, y_train.ravel())
    train_score = dt_model.score(X_train, y_train)
    test_score = dt_model.score(X_test, y_test)

    if plotting:
        print("- train score:\t"+str(train_score)) 
        print("- test score:\t"+str(test_score))  

    return test_score, train_score

    ################# K Neighbors classifier ##############################

def k_neighbors_class(data_x, data_y, feature_names, k, plotting):

    X_train, X_test, y_train, y_test = normalize_data(data_x, data_y)

    n_neighbors = 5
    knr_model = neighbors.KNeighborsClassifier(n_neighbors, weights="distance", algorithm="kd_tree")
    knr_model.fit(X_train, y_train.ravel())
    train_score = knr_model.score(X_train, y_train)
    test_score = knr_model.score(X_test, y_test)

    if plotting:
        print("- train score:\t"+str(train_score)) 
        print("- test score:\t"+str(test_score))  

    return test_score, train_score
