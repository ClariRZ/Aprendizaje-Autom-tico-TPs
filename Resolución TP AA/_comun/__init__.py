from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

SEMILLA=42
RES=Path(__file__).resolve().parents[1]
ROOT=RES.parent
DATA=ROOT/"00_base_de_datos"/"dataset_TP1_2026.csv"
OUT=RES/"resultados"
OUT.mkdir(exist_ok=True)

def cargar():
    if not DATA.exists():
        raise FileNotFoundError("Primero ejecutar 00_base_de_datos/reconstruir_dataset.py")
    return pd.read_csv(DATA)

def xy(df,binario=True):
    X=df.drop(columns="target").copy()
    y=(df.target=="Desertor").astype(int) if binario else df.target.copy()
    return X,y

def columnas(X):
    num=X.select_dtypes(include=np.number).columns.tolist()
    cat=X.select_dtypes(exclude=np.number).columns.tolist()
    return num,cat

def pre(X,escalar=True):
    num,cat=columnas(X)
    pasos_num=[("imputar",SimpleImputer(strategy="median"))]
    if escalar: pasos_num.append(("escalar",StandardScaler()))
    return ColumnTransformer([
        ("num",Pipeline(pasos_num),num),
        ("cat",Pipeline([("imputar",SimpleImputer(strategy="most_frequent")),
                         ("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ])


def cargar_eda_train():
    """EDA original: reservar test antes de cualquier cálculo descriptivo o supervisado."""
    df = cargar()
    indices_train, _ = train_test_split(
        np.arange(len(df)), test_size=0.20, stratify=df["target"], random_state=SEMILLA
    )
    return df.iloc[indices_train].copy()
