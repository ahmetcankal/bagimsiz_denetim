from unittest import result
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
testresults=[]
resultsd=dict()
tsutun1 = ['Model', 'Değişken_sayisi','cv_ort','trainscore','testscore','f1','precision','recall','rocauc']
gstsutun=['model','variable_count','best_mean_score_accuracy','best_params','best_mean_precision','best_mean_recall','best_mean_f1']
modelname="svm_linear"
Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  modelname)
#print (Xtum.describe(include='all'))
for k in range(3,14,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  
    #test,train,result_models,tabledict = models.azmodeluygula(data_x, data_y, 
                                            #yenifeatures, k, isPlotting)
    result_models,test_models = models.azgridcv(data_x, data_y, 
                                            yenifeatures, k, isPlotting)
   # test,train = models.azsupport_vector_machine_linear(data_x, data_y, 
   #                                         yenifeatures, k, isPlotting)
    #results.append([k,test,train])
    #results.append(result_models)
    results.extend(result_models)
    testresults.extend(test_models)
   
#all_results[modelname] = results
#utils.stop_timer(initial, modelname)
#utils.save_results(results, modelname)
#utils.plot_results(results, False, modelname)

#df = pd.DataFrame(results,columns=tsutun)
#df.to_csv(r'Results/all_results.csv', index=False)

gsresults = pd.DataFrame(results)
gsresults.columns=['model','variable_count','train_accuracy','best_params','train_precision','train_recall','train_f1','train_roc_auc','train_kappa']
gsresults.to_csv(r'Results/traings_results.csv', index=False)

testresults = pd.DataFrame(testresults)
testresults.columns=['model','variable_count','test_accuracy','test_params','test_precision','test_recall','test_f1','test_roc_auc','test_kappa']
testresults.to_csv(r'Results/testgs_results.csv', index=False)

