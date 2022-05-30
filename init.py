import pandas as pd
sonuclar=[]
import models
import utils


all_results={}
total_tests = 0
isPlotting = False
utils.create_folder("Results") 





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





#utils.plot_all_results(all_results)

