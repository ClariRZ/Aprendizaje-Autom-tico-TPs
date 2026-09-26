from pathlib import Path
import sys
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from _comun import cargar_eda_train, OUT
from eda_utils import cramers_v

BASE = OUT / "eda_exploratorio"
TAB = BASE / "tablas"
TAB.mkdir(parents=True, exist_ok=True)

df = cargar_eda_train()
pred = df.drop(columns="target")

# Numéricas
num = pred.select_dtypes(include=np.number)
corr = num.corr(method="spearman")
corr.to_csv(TAB / "15_correlacion_spearman.csv")

pairs = []
cols = corr.columns.tolist()
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        rho = corr.iloc[i,j]
        pairs.append({
            "variable_1": cols[i],
            "variable_2": cols[j],
            "rho_spearman": rho,
            "abs_rho": abs(rho),
        })
pairs = pd.DataFrame(pairs).sort_values("abs_rho", ascending=False)
pairs.to_csv(TAB / "16_pares_numericos.csv", index=False)

# Categóricas
cat_cols = pred.select_dtypes(exclude=np.number).columns.tolist()
mat = pd.DataFrame(np.eye(len(cat_cols)), index=cat_cols, columns=cat_cols)
cat_pairs = []
for i in range(len(cat_cols)):
    for j in range(i+1, len(cat_cols)):
        v = cramers_v(df[cat_cols[i]].astype(str), df[cat_cols[j]].astype(str))
        mat.iloc[i,j] = v
        mat.iloc[j,i] = v
        cat_pairs.append({
            "variable_1": cat_cols[i],
            "variable_2": cat_cols[j],
            "cramers_v": v,
        })
mat.to_csv(TAB / "17_cramers_v_entre_categoricas.csv")
cat_pairs = pd.DataFrame(cat_pairs).sort_values("cramers_v", ascending=False)
cat_pairs.to_csv(TAB / "18_pares_categoricos.csv", index=False)

print("PARES NUMÉRICOS")
print(pairs.head(15).to_string(index=False))
print("\nPARES CATEGÓRICOS")
print(cat_pairs.head(15).to_string(index=False))
