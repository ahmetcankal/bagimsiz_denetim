
from pyexpat import model
from trace import CoverageResults
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, ConfusionMatrixDisplay,f1_score,precision_score,accuracy_score,recall_score,roc_auc_score,cohen_kappa_score
from sklearn.metrics import make_scorer
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
from sklearn.model_selection import GridSearchCV
import seaborn as sns


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

#şimdilik kullanılmıyor.
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
        'svm':{
            'model':svm.SVC(gamma='auto'),
            'params':{'C': [0.1,1,5,10,15,20,100,1000],'kernel': ['rbf','linear','poly','sigmoid'],'degree':[3,8] }
        }
        ,
        'random_forest':{ 
            'model':RandomForestClassifier(),
            'params':{'n_estimators':[1,5,10,50,100] ,'max_depth':[4,5,6,7,8,9,10],'min_samples_leaf':[2,10,20],'min_samples_split':[10,15,20],'criterion':["gini","entropy"],'max_features':['auto', 'sqrt'] }
         },
         'DT':{
            'model':DecisionTreeClassifier(),
            'params':{'min_samples_leaf': range(1,4),'min_samples_split': [2, 3, 4],'criterion':['gini', 'entropy'], 'max_depth':[2,4,6,8,10,12],'max_features': ['auto', 'sqrt', 'log2']
            }

         }
#DT   leafnodelistrange=100 , samples range=5 maxdepth=,4,5,6,7,8,9,10,11,12 ekle
#rf estimator=1,5,10,50,100 depth =4,5,6,7,8,9,10
#svm C=0.1,1,5,10,15,20,100,

    }
    #model paramları bitti

    models=[]
    #models.append(('SVM',svm.SVC(gamma='auto')))
    
    #models.append(('SVMP',svm.SVC(C=param_C_poly, degree=3, kernel="poly")))
    #models.append(('SVMRBF',svm.SVC(C=param_C_rbf, kernel="rbf")))
    #models.append(('SVMSig',svm.SVC(C=param_C_sigmoid, kernel="sigmoid")))
    metrikler = { "Accuracy": make_scorer(accuracy_score),'Precision':make_scorer(precision_score),'recall':make_scorer(recall_score),'f1':make_scorer(f1_score)}
    #print(models)
    X_train, X_test, y_train, y_test,X_std,Y_std = aznormalize_data(data_x, data_y)
    Y_std = Y_std.ravel()
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    results=dict()
    names=[]
    table1=[]
    tsutun = ['Model', 'Değişken_sayisi','cv_ort','trainscore','testscore','f1','precision','recall','rocauc']
    tabledict=dict()
    scores=[]
    testscores=[]
    for model_name,mp in model_params.items():
        clf=GridSearchCV(estimator=mp['model'],param_grid=mp['params'],cv=5,scoring=metrikler,return_train_score=True,refit="Accuracy",n_jobs= -1, verbose= 2 )
        #train verisi 
        modelim = clf.fit(X_train,y_train)
        train_accuracy = clf.score(X_train,y_train)
        #clf.decision_function()
        #clf.get_params()
        print(modelim.best_params_)
       # cmtrain = confusion_matrix(X_train,y_train)   
        #scores.append (clf.score(X_train,y_train))
        clf_testpredicted = clf.predict(X_test)
        
        #train verileri modele veriliyor
        clf_trainpredicted=clf.predict(X_train)
       
        cm = confusion_matrix(y_test,clf_testpredicted)       
        cmtrain=confusion_matrix(y_train,clf_trainpredicted)






        
        #tüm veri için
        #clf.fit(X_std,Y_std)

        # test model verileri alttaki gibi alınıp diziye aktarılabilir.
        print(classification_report(y_test,clf_testpredicted))

        print(classification_report(y_train,clf_trainpredicted))

        # print(f"Accuracy: {round(accuracy_score(y_test, clf_testpredicted), 2)}")
        # print(f"Precision: {round(precision_score(y_test, clf_testpredicted,average='weighted'), 2)}")
        # print(f"Recall: {round(recall_score(y_test, clf_testpredicted,average='weighted'), 2)}")
        # print(f"F1_score: {round(f1_score(y_test, clf_testpredicted,average='weighted'), 2)}")



        
       #test değerleri

        


        test_accuracy = round(accuracy_score(y_test, clf_testpredicted), 6)
        # test_precision = round(precision_score(y_test, clf_testpredicted,average='weighted'), 6)
        # test_recall = round(recall_score(y_test, clf_testpredicted,average='weighted'), 6)
        # test_f1 = round(f1_score(y_test, clf_testpredicted,average='weighted'), 6)
        test_precision,test_recall, test_f1, support_ = metrics.precision_recall_fscore_support(y_test, clf_testpredicted, average='weighted')

        #test_roc_auc = round(roc_auc_score(y_test, clf_testpredicted,average='weighted'), 6)
        test_kappa = round(cohen_kappa_score(y_test, clf_testpredicted),6)



        testscores.append(
            {
            'model':model_name,
            'variable_count':k,
            'test_accuracy':test_accuracy,
            'test_params':clf.best_params_,
            'test_precision':test_precision,
            'test_recall':test_recall,
            'test_f1':test_f1,
            'test_auc':test_accuracy          
        })

        testgsresults=pd.DataFrame(testscores,columns=['model','variable_count','test_accuracy','test_params','test_precision','test_recall','test_f1','test_auc'])
        
        testgsresults.to_csv('testgsresult_'+str(k)+'.csv')



                #train değerleri
        i = clf.best_index_
        mean_precision = clf.cv_results_['mean_test_Precision'][i]
        mean_recall = clf.cv_results_['mean_test_recall'][i]   
        mean_f1 = clf.cv_results_['mean_test_f1'][i]


        train_accuracy = round(accuracy_score(y_train, clf_trainpredicted), 6)
        
        
        train_precision,train_recall, train_f1, support_ = metrics.precision_recall_fscore_support(y_train, clf_trainpredicted, average='weighted')

        #train_roc_auc = round(roc_auc_score(y_train, clf_trainpredicted,average='weighted'), 6)
        train_kappa = round(cohen_kappa_score(y_train, clf_trainpredicted),6)

        #train score değerleri  düzenlendi elde edildi
        scores.append({
            'model':model_name,
            'variable_count':k,
            'train_accuracy':train_accuracy,
            'best_params':clf.best_params_,
            'train_precision':train_precision,
            'train_recall':train_recall,
            'train_f1':train_f1,
            'train_auc':train_accuracy
            
        })
        gsresults=pd.DataFrame(scores,columns=['model','variable_count','train_accuracy','best_params','train_precision','train_recall','train_f1','train_roc_auc'])

        print(gsresults)
        df2=pd.DataFrame(clf.cv_results_)
        print(df2)
        df2.to_csv(model_name+'model1.csv')
        gsresults.to_csv('traingsresult_'+str(k)+'.csv')
        #df2=df2.sort_values("accuracy")
        

        #clf_testpredicted=clf.predict(X_test)
        #confusion_matrix(y_test,clf_testpredicted)  


        #predy=clf.predict(X_test)
        #print(classification_report(y_test,predy))
           

        #cvresults_acc=clf.best_score_
    trainlistresult = gsresults.values.tolist()
    testlistresult = testgsresults.values.tolist()

    return trainlistresult,testlistresult

# def duzenlenecek():       
#         kfold=KFold(n_splits=10,random_state=7,shuffle=True)
#         cvresults_acc=cross_val_score(model,X_std,Y_std,cv=10,scoring="accuracy")
#         cvresults_f1=cross_val_score(model,X_std,Y_std,cv=10,scoring="f1_macro")
#         cvresults_preci=cross_val_score(model,X_std,Y_std,cv=10,scoring="precision")
#         cvresults_recall=cross_val_score(model,X_std,Y_std,cv=10,scoring="recall")
#         cvresults_rocauc=cross_val_score(model,X_std,Y_std,cv=10,scoring="roc_auc")
#         results[name]=(cvresults_acc.mean(),cvresults_acc.std(),cvresults_f1.mean(),cvresults_preci.mean(),cvresults_recall.mean(),cvresults_rocauc.mean())
#         names.append(name)

#         model=model.fit(X_train,y_train)
#         train_score = model.score(X_train, y_train)
#         test_score = model.score(X_test, y_test)
#         table1.append([name,k,cvresults_acc.mean(),train_score,test_score,cvresults_f1.mean(),cvresults_preci.mean(),cvresults_recall.mean(),cvresults_rocauc.mean()])
#         tabledict[name]=(name,k,cvresults_acc.mean(),train_score,test_score)
#     print()
#     print("name   results.mean      result.std ")

#     #for key,value in results.items():
#     #    print(key,value)

#     for sonuc in table1:
#         print(sonuc)
#     for key,value in tabledict.items():
#         print(key,value)
        
# return cvresults