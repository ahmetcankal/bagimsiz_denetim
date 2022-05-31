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

Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  "linear_svr")
 
for k in range(3,13,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  


    test,train = models.azsupport_vector_machine_linear(data_x, data_y, 
                                            yenifeatures, k, isPlotting)
    results.append([k,test,train])
all_results["linear_svr"] = results
utils.stop_timer(initial, "linear_svr")
utils.save_results(results, "linear_svr")
utils.plot_results(results, False, "linear_svr")


######################## rbf support vector machine method #################################
initial = utils.start_timer()
results=[]

Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  "svr_rbf")
 
for k in range(3,13,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  


    test,train = models.azsupport_vector_machine_rbf(data_x, data_y, 
                                            yenifeatures, k, isPlotting)
    results.append([k,test,train])
all_results["svr_rbf"] = results
utils.stop_timer(initial, "svr_rbf")
utils.save_results(results, "svr_rbf")
utils.plot_results(results, False, "svr_rbf")


######################## poly support vector machine method #################################
initial = utils.start_timer()
results=[]

Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  "svr_poly")
 
for k in range(3,13,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  


    test,train = models.azsupport_vector_machine_poly(data_x, data_y, 
                                            yenifeatures, k, isPlotting)
    results.append([k,test,train])
all_results["svr_poly"] = results
utils.stop_timer(initial, "svr_poly")
utils.save_results(results, "svr_poly")
utils.plot_results(results, False, "svr_poly")


######################## dt  #################################
initial = utils.start_timer()
results=[]

Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  "dt")
 
for k in range(3,13,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  
    test,train = models.azDecision_Tree_class(data_x, data_y, 
                                                  yenifeatures, k, isPlotting)


    results.append([k,test,train])
all_results["dt"] = results
utils.stop_timer(initial, "dt")
utils.save_results(results, "dt")
utils.plot_results(results, False, "dt")


######################## random forest  #################################
initial = utils.start_timer()
results=[]

Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  "random forest")
 
for k in range(3,13,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  
    test,train = models.azRandom_forest(data_x, data_y, 
                                                  yenifeatures, k, isPlotting)


    results.append([k,test,train])
all_results["random forest"] = results
utils.stop_timer(initial, "random forest")
utils.save_results(results, "random forest")
utils.plot_results(results, False, "random forest")

