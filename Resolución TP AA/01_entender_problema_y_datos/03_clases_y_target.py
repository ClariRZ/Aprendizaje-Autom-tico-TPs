from pathlib import Path
import sys
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from _comun import cargar_eda_train, OUT
from eda_utils import CLASES, ACADEMICAS, eta_squared, cramers_v, smd_binario

BASE = OUT / "eda_exploratorio"
TAB = BASE / "tablas"
TAB.mkdir(parents=True, exist_ok=True)

df = cargar_eda_train()
pred = df.drop(columns="target")
num_cols = pred.select_dtypes(include=np.number).columns.tolist()
cat_cols = pred.select_dtypes(exclude=np.number).columns.tolist()

# Numéricas por las tres clases
rows = []
for c in num_cols:
    for clase in CLASES:
        x = df.loc[df["target"] == clase, c].dropna()
        rows.append({
            "variable": c,
            "clase": clase,
            "n": len(x),
            "media": x.mean(),
            "mediana": x.median(),
            "q25": x.quantile(.25),
            "q75": x.quantile(.75),
            "pct_ceros": 100 * x.eq(0).mean(),
        })
num3 = pd.DataFrame(rows)
num3.to_csv(TAB / "08_numericas_por_tres_clases.csv", index=False)

rank_num = []
for c in num_cols:
    sub = num3[num3["variable"] == c]
    rank_num.append({
        "variable": c,
        "eta2_tres_clases": eta_squared(df[c], df["target"]),
        "rango_mediana": sub["mediana"].max() - sub["mediana"].min(),
        "max_gap_ceros_pp": sub["pct_ceros"].max() - sub["pct_ceros"].min(),
    })
rank_num = pd.DataFrame(rank_num).sort_values("eta2_tres_clases", ascending=False)
rank_num.to_csv(TAB / "09_resumen_numericas_tres_clases.csv", index=False)

# Categóricas por las tres clases
global_p = df["target"].value_counts(normalize=True)
cat_rows = []
for c in cat_cols:
    for categoria, s in df.groupby(c, dropna=False):
        n = len(s)
        vc = s["target"].value_counts()
        row = {"variable": c, "categoria": str(categoria), "n": n}
        for clase in CLASES:
            p = vc.get(clase, 0) / n
            row[f"pct_{clase}"] = 100 * p
            row[f"lift_{clase}"] = p / global_p[clase]
        cat_rows.append(row)
cat3 = pd.DataFrame(cat_rows)
cat3.to_csv(TAB / "10_categorias_por_tres_clases.csv", index=False)

assoc_cat = pd.DataFrame({
    "variable": cat_cols,
    "cramers_v_target3": [cramers_v(df[c].astype(str), df["target"]) for c in cat_cols],
}).sort_values("cramers_v_target3", ascending=False)
assoc_cat.to_csv(TAB / "11_asociacion_categoricas_target3.csv", index=False)

# Ceros vs nivel positivo en variables académicas
acad = []
for c in ACADEMICAS:
    for clase in CLASES:
        x = df.loc[df["target"] == clase, c].dropna()
        xp = x[x > 0]
        acad.append({
            "variable": c,
            "clase": clase,
            "n": len(x),
            "pct_ceros": 100 * x.eq(0).mean(),
            "n_positivos": len(xp),
            "mediana_positivos": xp.median() if len(xp) else np.nan,
            "media_positivos": xp.mean() if len(xp) else np.nan,
        })
pd.DataFrame(acad).to_csv(TAB / "12_academicas_ceros_y_positivos.csv", index=False)

# Puente a la tarea binaria
y = (df["target"] == "Desertor").astype(int)

bin_num = []
for c in num_cols:
    x0 = df.loc[y == 0, c]
    x1 = df.loc[y == 1, c]
    bin_num.append({
        "variable": c,
        "mediana_no_desertor": x0.median(),
        "mediana_desertor": x1.median(),
        "pct_ceros_no_desertor": 100 * x0.eq(0).mean(),
        "pct_ceros_desertor": 100 * x1.eq(0).mean(),
        "smd": smd_binario(df[c], y),
    })
bin_num = pd.DataFrame(bin_num)
bin_num["abs_smd"] = bin_num["smd"].abs()
bin_num = bin_num.sort_values("abs_smd", ascending=False)
bin_num.to_csv(TAB / "13_puente_binario_numericas.csv", index=False)

global_dropout = y.mean()
bin_cat = []
for c in cat_cols:
    tmp = pd.DataFrame({"categoria": df[c].astype(str), "desertor": y})
    agg = tmp.groupby("categoria")["desertor"].agg(["size", "mean"]).reset_index()
    for _, r in agg.iterrows():
        bin_cat.append({
            "variable": c,
            "categoria": r["categoria"],
            "n": int(r["size"]),
            "tasa_desercion": r["mean"],
            "diferencia_pp_vs_global": 100 * (r["mean"] - global_dropout),
        })
bin_cat = pd.DataFrame(bin_cat)
bin_cat["abs_diferencia_pp"] = bin_cat["diferencia_pp_vs_global"].abs()
bin_cat = bin_cat.sort_values(["abs_diferencia_pp","n"], ascending=[False,False])
bin_cat.to_csv(TAB / "14_puente_binario_categorias.csv", index=False)

print("TOP NUMÉRICAS — TRES CLASES")
print(rank_num.head(11).to_string(index=False))
print("\nCATEGÓRICAS — TRES CLASES")
print(assoc_cat.to_string(index=False))
print("\nPUENTE BINARIO — NUMÉRICAS")
print(bin_num.head(11).to_string(index=False))
