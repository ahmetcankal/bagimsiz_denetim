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
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.preprocessing import MinMaxScaler

def loaddata(k):
    data = pd.read_csv('erdemveri01.csv',sep=";")
    
    data_y=data.loc[:,"Y"]
    data_x=data.loc[:,"CO":"ROIC"]
    feature_names = list(data_x.columns)
    Selected_feature_names=fs(data_x,data_y,k)
    data_x = data_x[data_x.columns.intersection(Selected_feature_names)]
    data_x = data_x.to_numpy()
    data_y = data_y.to_numpy()
 
 
    return data_x,data_y,feature_names


def fs(X,y,_k):
    scalerx = MinMaxScaler().fit(X)
    Xs = scalerx.fit_transform(X)
    scalery = MinMaxScaler()
    #Y_std = y.reshape(-1,1)
    selector=SelectKBest(score_func=chi2,k=_k)
    model= selector.fit(Xs,y)
    Selected_feature_names=X.columns[model.get_support()]
    skor=model.scores_
    zipped=zip(Selected_feature_names,skor)
    zipped=sorted(zipped, key=lambda x: x[1],reverse=True)

    for f,s in zipped:
        print(f,s)

    return Selected_feature_names

  
def loadfirma(ulke):
     data = pd.read_csv('erdemveri01.csv', sep=";")
     #data = data[data['hisseadi'] == hisseadi] tüm verileri tek csv dosyasında birleştirirsek kullanabiliriz.
     data_y=data.loc[:,"Y"]
     data_ysinif=data.loc[:,"Y"]
     columnfeaturesx = list(data.columns)
     data_x=data.loc[:,"CO":"ROIC"]
     columnfeaturesx = list(data_x.columns)
     #data_x = data.drop(columns=data.columns[0], axis=1) #bastaki geo zaman 20001Q şeklindeki sütun kaldırılıyor
     #data_x = data_x.drop(columns=data.columns[-1], axis=1) #sondaki ülkeadi şeklindeki sütun kaldırılıyor
     #data_x = data_x.drop(columns=data.columns[-1], axis=1) #sondaki y  şeklindeki bağımlı değişken sütunuda  kaldırılıyor
     #data_x = data_x.drop(columns=data.columns[0], axis=1)
     data_x = data_x.to_numpy() #panda dataframe verileri numpy dizisine çevrildi
     data_y = data_y.to_numpy()
     data_ysinif = data_ysinif.to_numpy()
     
     return data_x,data_y,columnfeaturesx,data_ysinif
