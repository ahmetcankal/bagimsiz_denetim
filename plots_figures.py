
import os, time, cmath, math 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import colorcet as cc
from sklearn.feature_selection import SelectKBest, chi2, f_regression,mutual_info_classif,f_classif,RFE
import seaborn as sns


acclist=[]
auclist=[]
cmlist=[]






def azfeatimportance(_file):
    data_tum = pd.read_csv(_file, sep=";")


    #print(data_tum)
    plt.figure(figsize=(12, 6))
    plt.xlabel('variables')
    plt.ylabel('score')
    plt.title('Feature Importances mutual_info_classif')
    palette = sns.color_palette(cc.glasbey, n_colors=13)
    sns.barplot(x=data_tum.score, y=data_tum.variables, orient='h',palette=palette)
    plt.savefig('Results/mutual_variable_score_table.png', dpi=300)
    return None
def azmodelvartrain(_file):
    data_tum = pd.read_csv(_file, sep=";")
    return None

def azmodelvartest(_file):
    df = pd.read_csv(_file, sep=";")
    #df.set_index("model", inplace=True)
    #print(df)
    tidy = df.stack().reset_index().rename(columns={"level_1": "model", 0: "test_accuracy"})
    #print(tidy)
    sns.lineplot(data=tidy[tidy["model"].isin(["3", "13"])], x="variable_count", y="test_accuracy")
    

    plt.savefig('Results/models_accuracy.png', dpi=300)

    return None