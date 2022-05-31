#veriler burdan yüklenecek
#df = pd.read_csv('/content/drive/MyDrive/00dosyalar/pyhton/dostumhissedegiskenler.csv')
# data = pd.read_csv('/content/drive/MyDrive/00dosyalar/pyhton/dostumhissedegiskenler.csv', sep=";")   #on_bad_lines='skip'
# data.head()
# data_y=data.loc[:,"hkapanis"]
# data_y.head()
# data_x = data.drop(columns=data.columns[-1], axis=1)
# data_x = data.drop(columns=data.columns[0], axis=1)

# data_x.head()

# x = data[data['hisseadi'] == 'ARCLK']
# x.info()


import os, time, cmath, math 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.feature_selection import SelectKBest, chi2, f_regression,mutual_info_classif,f_classif


acclist=[]
auclist=[]
cmlist=[]



def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print("New directory for results created.") 
    return None


def start_timer():
    return time.time()
    

def stop_timer(initial, method_name):
    end = time.time() - initial
    print( "Time elapsed for method "+method_name+": " + str(format(end, '.0f'))+ " sec.")
    return end

def azload_excel_data(file_name,sheet_name, method_name):
    #data = pd.read_excel(file_name,sheet_name=sheet_name) 
    data = pd.read_csv('data/erdemveri01.csv', sep=";")
    data_y=data.loc[:,"Y"]
    data_x=data.loc[:,"CO":"ROIC"]
    feature_names = list(data_x.columns)

    Selected_feature_names=azsave_k_highest_scores(data_x,data_y, method_name)
    #data_x = data_x[data_x.columns(Selected_feature_names)]
    data_x = data_x.loc[:,Selected_feature_names]
    
    data_x = data_x.to_numpy()
    data_y = data_y.to_numpy()
    
    return data_x,data_y,Selected_feature_names


def azsave_k_highest_scores(data_x,data_y, method_name):
    selector=SelectKBest(score_func=mutual_info_classif, k=13)
    model= selector.fit(data_x,data_y)
    selected_feature_names=data_x.columns[model.get_support()]
    scores=model.scores_
    zipped=zip(selected_feature_names,scores)
    zipped=sorted(zipped, key=lambda x: x[1],reverse=True)
    
    df = pd.DataFrame(zipped, columns=["variables","score"])
    create_folder("Results/"+method_name+'/')
    df.to_csv('Results/'+method_name+'/'+'_tumvariables.csv', index=False)
    # for feature,score in zipped:
    #     print(feature,score)
    selected_feature_names=[]
    for f,s in zipped:
        selected_feature_names.append(f)

    return selected_feature_names

def load_excel_data(file_name,sheet_name,k, method_name):
    #data = pd.read_excel(file_name,sheet_name=sheet_name) 
    data = pd.read_csv('data/erdemveri01.csv', sep=";")
    data_y=data.loc[:,"Y"]
    data_x=data.loc[:,"CO":"ROIC"]
    feature_names = list(data_x.columns)
    Selected_feature_names=save_k_highest_scores(data_x,data_y,k, method_name)
    data_x = data_x[data_x.columns.intersection(Selected_feature_names)]
    data_x = data_x.to_numpy()
    data_y = data_y.to_numpy()
    
    return data_x,data_y,feature_names


def save_k_highest_scores(data_x,data_y,_k, method_name):
    selector=SelectKBest(score_func=f_classif, k=_k)
    model= selector.fit(data_x,data_y)
    selected_feature_names=data_x.columns[model.get_support()]
    scores=model.scores_
    zipped=zip(selected_feature_names,scores)
    zipped=sorted(zipped, key=lambda x: x[1],reverse=True)
    
    df = pd.DataFrame(zipped, columns=["variables","score"])
    create_folder("Results/"+method_name+'/')
    df.to_csv('Results/'+method_name+'/'+str(_k)+'_variables.csv', index=False)
    # for feature,score in zipped:
    #     print(feature,score)
    selected_feature_names=[]
    for f,s in zipped:
        selected_feature_names.append(f)

    return selected_feature_names
 

def plot_f_importance(coef, names,_k, method_name):
    imp = coef[0]
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
   
    fig = plt.gcf()
    fig.set_size_inches((15, 15), forward=False)
    create_folder("Results/"+method_name)
    plt.savefig('Results/'+method_name+'/'+str(_k)+'_variables.png', dpi=300)
    plt.close()
    
    return None


def plot_results(results, show_plot, method_name):
    plt.figure(figsize=(5, 2.7), layout='constrained')
    np_results = np.array(results)
    variables = np_results[:,0].astype(int)
    test_results = np_results[:,1]
    train_results = np_results[:,2]

    rlen = len(test_results)-1
    variables_order = -1
    max_score = 0
    for i in range(0, rlen):
        score = test_results[i]
        if score > max_score:
            max_score = score
            variables_order = i

    plt.scatter(variables[variables_order], 
                test_results[variables_order], c="red")
    plt.annotate(str(format(test_results[variables_order], '.4f')),
                ( variables[variables_order], test_results[variables_order]))
     
    #plt.scatter(variables[variables_order], 
    #           train_results[variables_order], c="blue")
 

    plt.plot(variables, test_results, label='Test')  
    plt.plot(variables, train_results, label='Train')   
    plt.xlabel('Variables')
    plt.ylabel('Score')
    plt.title("Variable Score Table for "+method_name)
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches((5, 5), forward=False)
    create_folder("Results")
    plt.savefig('Results/'+method_name+'_variable_score_table.png', dpi=300)

    if show_plot:
        plt.show()

    plt.close()

    return None


def plot_all_results(all_results):
 
    axesSize = math.ceil(math.sqrt(len(all_results)))
    figure, axis = plt.subplots(axesSize, axesSize)

    axis_x = 0
    axis_y = 0
    for model_name, rslt in all_results.items(): 

        np_results = np.array(rslt)
        variables = np_results[:,0].astype(int)
        test_results = np_results[:,1]
        train_results = np_results[:,2]
     
        rlen = len(test_results)-1
        variables_order = -1
        max_score = 0
        for i in range(0, rlen):
            score = test_results[i]
            if score > max_score:
                max_score = score
                variables_order = i
        
        axis[axis_x, axis_y].scatter(variables[variables_order], 
                                test_results[variables_order], c="red")
        axis[axis_x, axis_y].annotate(str(format(test_results[variables_order], 
                 '.4f')),
                (variables[variables_order], test_results[variables_order]))
 
        axis[axis_x, axis_y].plot(variables, test_results, label='Test')  
        axis[axis_x, axis_y].plot(variables, train_results, label='Train')   
        # axis[axis_x, axis_y].xlabel('Variables')
        # axis[axis_x, axis_y].ylabel('Score')
        axis[axis_x, axis_y].set_title(model_name)
        # axis[axis_x, axis_y].legend()
        axis_y = axis_y + 1
        if(axis_y==axesSize):
            axis_y = 0
            axis_x = axis_x + 1

   
    figure.set_size_inches((axesSize*3, axesSize*3), forward=False)
    create_folder("Results")
    plt.savefig('Results/all_score_tables.png', dpi=300)
    plt.close()

    return None


def save_results(results, method_name):
    df = pd.DataFrame(results, columns=["variables","test","train"])
    df.to_csv(r'Results/'+method_name+'_results.csv', index=False)

    return None
    
def save_all_results(all_results):
    df = pd.DataFrame(all_results, columns=["variables","test","train","model"])
    df.to_csv(r'Results/all_results.csv', index=False)

    return None