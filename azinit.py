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
modelname="svm_linear"
Xtum, data_y, feature_names = utils.azload_excel_data(
        'data/data.xlsx', 0,  modelname)
 
for k in range(8,14,1):
    data_x=Xtum[:,:k]
    
    yenifeatures=feature_names[0:k]  
    test,train,table1,tabledict = models.azmodeluygula(data_x, data_y, 
                                            yenifeatures, k, isPlotting)

   # test,train = models.azsupport_vector_machine_linear(data_x, data_y, 
   #                                         yenifeatures, k, isPlotting)
    #results.append([k,test,train])
    results.append(table1)
#all_results[modelname] = results
#utils.stop_timer(initial, modelname)
#utils.save_results(results, modelname)
#utils.plot_results(results, False, modelname)
df = pd.DataFrame(results)
df.to_csv(r'Results/all_results.csv', index=False)

