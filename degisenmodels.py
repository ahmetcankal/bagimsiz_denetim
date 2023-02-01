 svmmodel=svmmodel.fit(X_train,y_train)

    train_score = svmmodel.score(X_train, y_train)
    test_score = svmmodel.score(X_test, y_test)

    y_pred=svmmodel.predict(X_test)
    confusion_matrix(y_test,y_pred)
    plot_confusion_matrix(svmmodel,X_test,y_test,cmap=plt.cm.Blues)
    plt.show()
    print(classification_report(y_test,y_pred))



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

############ nu support vector Machines ########################
def azsupport_vector_machine_linear(data_x, data_y, feature_names, k, plotting):

    X_train, X_test, y_train, y_test,X_std,Y_std = aznormalize_data(data_x, data_y)
    svmmodel = svm.SVC(gamma=0.001,  C=param_C_linear, kernel = 'linear') 
    #svmodel=svmmodel.fit(X_train, y_train.ravel())
    
    scores=cross_val_score(svmmodel,X_std,Y_std,cv=10,scoring="accuracy")
    print(scores.mean())
    svmmodel=svmmodel.fit(X_train,y_train)

    train_score = svmmodel.score(X_train, y_train)
    test_score = svmmodel.score(X_test, y_test)

    y_pred=svmmodel.predict(X_test)
    confusion_matrix(y_test,y_pred)
    plot_confusion_matrix(svmmodel,X_test,y_test,cmap=plt.cm.Blues)
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











initten kalanlar


######################## rbf support vector machine method #################################
initial = utils.start_timer()
results=[]
modelname="svm_rbf"
Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  modelname)

for k in range(3,14,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  


    test,train = models.azsupport_vector_machine_rbf(data_x, data_y, 
                                            yenifeatures, k, isPlotting)
    results.append([k,test,train])
all_results[modelname] = results
utils.stop_timer(initial, modelname)
utils.save_results(results, modelname)
utils.plot_results(results, False, modelname)


######################## poly support vector machine method #################################
initial = utils.start_timer()
results=[]
modelname="svm_poly"
Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  modelname)
 
for k in range(8,14,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  


    test,train = models.azsupport_vector_machine_poly(data_x, data_y, 
                                            yenifeatures, k, isPlotting)
    results.append([k,test,train])
all_results[modelname] = results
utils.stop_timer(initial, modelname)
utils.save_results(results, modelname)
utils.plot_results(results, False, modelname)


######################## dt  #################################
initial = utils.start_timer()
results=[]
modelname="Decision Tree"
Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  modelname)
 
for k in range(1,14,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  
    test,train = models.azDecision_Tree_class(data_x, data_y, 
                                                  yenifeatures, k, isPlotting)


    results.append([k,test,train])
all_results[modelname] = results
utils.stop_timer(initial, modelname)
utils.save_results(results, modelname)
utils.plot_results(results, False, modelname)


######################## random forest  #################################
initial = utils.start_timer()
results=[]
modelname="Random Forest"
Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  modelname)
 
for k in range(1,14,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  
    test,train = models.azRandom_forest(data_x, data_y, 
                                                  yenifeatures, k, isPlotting)


    results.append([k,test,train])
all_results[modelname] = results
utils.stop_timer(initial, modelname)
utils.save_results(results, modelname)
utils.plot_results(results, False, modelname)


