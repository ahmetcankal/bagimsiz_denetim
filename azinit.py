import pandas as pd
sonuclar=[]
import models
import utils



all_results={}
total_tests = 0
isPlotting = False
utils.create_folder("Results") 

acclist=[]
auclist=[]
cmlist=[]




######################## linear support vector machine method #################################
initial = utils.start_timer()
results=[]
modelname="svm_linear"
Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  modelname)
 
for k in range(3,14,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  


    test,train = models.azsupport_vector_machine_linear(data_x, data_y, 
                                            yenifeatures, k, isPlotting)
    results.append([k,test,train])
all_results[modelname] = results
utils.stop_timer(initial, modelname)
utils.save_results(results, modelname)
utils.plot_results(results, False, modelname)


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
 
for k in range(3,14,1):
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
 
for k in range(3,14,1):
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
 
for k in range(3,14,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  
    test,train = models.azRandom_forest(data_x, data_y, 
                                                  yenifeatures, k, isPlotting)


    results.append([k,test,train])
all_results[modelname] = results
utils.stop_timer(initial, modelname)
utils.save_results(results, modelname)
utils.plot_results(results, False, modelname)

