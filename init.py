import pandas as pd
sonuclar=[]
from models import main

for k in range(5,36,1):
    test,train = main(k)
    sonuclar.append([k,test,train])

print(sonuclar)

df = pd.DataFrame(sonuclar, columns=["degisken","test","train"])
df.to_csv(r'list.csv', index=False)

