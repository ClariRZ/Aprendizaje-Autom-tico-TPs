from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
R=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(R))
from _comun import *
df=cargar(); X,y=xy(df,True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEMILLA)
p=pre(Xtr,True); Xt=p.fit_transform(Xtr); Xv=p.transform(Xte)
print("train",Xtr.shape,"test",Xte.shape,"transformadas",Xt.shape[1])
print("prevalencia",y.mean(),ytr.mean(),yte.mean())
