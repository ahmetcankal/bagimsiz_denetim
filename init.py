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
for k in range(3,13,1):
    data_x, data_y, feature_names = utils.load_excel_data(
        'data/data.xlsx', 0, k, "linear_svr")
    test,train = models.support_vector_machine_linear(data_x, data_y, 
                                            feature_names, k, isPlotting)
    results.append([k,test,train])
all_results["linear_svr"] = results
utils.stop_timer(initial, "linear_svr")
utils.save_results(results, "linear_svr")
utils.plot_results(results, False, "linear_svr")


######################## poly support vector machine method #################################
initial = utils.start_timer()
results=[]
for k in range(3,13,1):
    data_x, data_y, feature_names = utils.load_excel_data(
        'Data/data.xlsx', 0, k, "svr_poly")
    test,train = models.support_vector_machine_poly(data_x, data_y, 
                                                  feature_names, k, isPlotting)
    results.append([k,test,train])
all_results["svr_poly"] = results
utils.stop_timer(initial, "svr_poly")
utils.save_results(results, "svr_poly")
utils.plot_results(results, False, "svr_poly")
 
######################## rbf support vector machine method #################################
initial = utils.start_timer()
results=[]
for k in range(3,13,1):
    data_x, data_y, feature_names = utils.load_excel_data(
        'Data/data.xlsx', 0, k, "svr_rbf")
    test,train = models.support_vector_machine_rbf(data_x, data_y, 
                                                  feature_names, k, isPlotting)
    results.append([k,test,train])
all_results["svr_rbf"] = results 
utils.stop_timer(initial, "svr_rbf")
utils.save_results(results, "svr_rbf")
utils.plot_results(results, False, "svr_rbf")

######################## sigmoid support vector machine method #################################
initial = utils.start_timer()
results=[]
for k in range(3,13,1):
    data_x, data_y, feature_names = utils.load_excel_data(
        'Data/data.xlsx', 0, k, "svr_sigmoid")
    test,train = models.support_vector_machine_sigmoid(data_x, data_y, 
                                                  feature_names, k, isPlotting)
    results.append([k,test,train])
all_results["svr_sigmoid"] = results 
utils.stop_timer(initial, "svr_sigmoid")
utils.save_results(results, "svr_sigmoid")
utils.plot_results(results, False, "svr_sigmoid")

######################## k neighbors class method #################################
initial = utils.start_timer()
results=[]
for k in range(3,13,1):
    data_x, data_y, feature_names = utils.load_excel_data(
        'Data/data.xlsx', 0, k, "k_neighbors")
    test,train = models.k_neighbors_class(data_x, data_y, 
                                                  feature_names, k, isPlotting)
    results.append([k,test,train])
all_results["k_neighbors"] = results
utils.stop_timer(initial, "k_neighbors")
utils.save_results(results, "k_neighbors")
utils.plot_results(results, False, "k_neighbors")

######################## decision t class method #################################
initial = utils.start_timer()
results=[]
for k in range(3,13,1):
    data_x, data_y, feature_names = utils.load_excel_data(
        'Data/data.xlsx', 0, k, "decision_tree")
    test,train = models.Decision_Tree_class(data_x, data_y, 
                                                  feature_names, k, isPlotting)
    results.append([k,test,train])
all_results["decision_tree"] = results
utils.stop_timer(initial, "decision_tree")
utils.save_results(results, "decision_tree")
utils.plot_results(results, False, "decision_tree")

######################## random forest class method #################################
initial = utils.start_timer()
results=[]
for k in range(3,13,1):
    data_x, data_y, feature_names = utils.load_excel_data(
        'Data/data.xlsx', 0, k, "random_forest")
    test,train = models.Random_forest(data_x, data_y, 
                                                  feature_names, k, isPlotting)
    results.append([k,test,train])
all_results["random_forest"] = results
utils.stop_timer(initial, "random_forest")
utils.save_results(results, "random_forest")
utils.plot_results(results, False, "random_forest")



######################## naive bayes class method #################################
initial = utils.start_timer()
results=[]
for k in range(3,13,1):
    data_x, data_y, feature_names = utils.load_excel_data(
        'Data/data.xlsx', 0, k, "naive_bayes")
    test,train = models.naivebayes(data_x, data_y, 
                                                  feature_names, k, isPlotting)
    results.append([k,test,train])
all_results["naive_bayes"] = results
utils.stop_timer(initial, "naive_bayes")
utils.save_results(results, "naive_bayes")
utils.plot_results(results, False, "naive_bayes")



#utils.plot_all_results(all_results)

