from pathlib import Path
import sys
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
ROOT = R.parent
sys.path.insert(0, str(R))
from _comun import cargar_eda_train, OUT
from eda_utils import momento

BASE = OUT / "eda_exploratorio"
TAB = BASE / "tablas"
TAB.mkdir(parents=True, exist_ok=True)

df = cargar_eda_train()
pred = df.drop(columns="target")

resumen = pd.DataFrame([{
    "filas": len(df),
    "columnas_totales": df.shape[1],
    "predictores": pred.shape[1],
    "faltantes_totales": int(df.isna().sum().sum()),
    "filas_identicas": int(df.duplicated().sum()),
    "numericas": int(pred.select_dtypes(include=np.number).shape[1]),
    "categoricas": int(pred.select_dtypes(exclude=np.number).shape[1]),
}])
resumen.to_csv(TAB / "01_resumen_base.csv", index=False)

target = df["target"].value_counts().rename_axis("clase").reset_index(name="n")
target["porcentaje"] = 100 * target["n"] / len(df)
target.to_csv(TAB / "02_target_tres_clases.csv", index=False)

dic_path = ROOT / "00_base_de_datos" / "diccionario_datos_TP1_2026.csv"
dic = pd.read_csv(dic_path)
obs = []
for c in pred.columns:
    obs.append({
        "Variable": c,
        "tipo_observado": "numérica" if pd.api.types.is_numeric_dtype(pred[c]) else "categórica",
        "n_unicos": int(pred[c].nunique(dropna=True)),
        "faltantes": int(pred[c].isna().sum()),
        "pct_ceros": 100 * pred[c].eq(0).mean() if pd.api.types.is_numeric_dtype(pred[c]) else np.nan,
        "momento": momento(c),
    })
obs = pd.DataFrame(obs)
dic_obs = dic[dic["Variable"] != "target"].merge(obs, on="Variable", how="right")
dic_obs.to_csv(TAB / "03_diccionario_observado.csv", index=False)

checks = [
    ("filas idénticas", int(df.duplicated().sum())),
    ("aprobadas S1 > inscriptas S1", int((df["Unidades_curriculares_1er_semestre_(aprobadas)"] > df["Unidades_curriculares_1er_semestre_(inscrito)"]).sum())),
    ("aprobadas S2 > inscriptas S2", int((df["Unidades_curriculares_2do_semestre_(aprobadas)"] > df["Unidades_curriculares_2do_semestre_(inscrito)"]).sum())),
    ("aprobadas S1 > 0 con evaluaciones S1 = 0", int(((df["Unidades_curriculares_1er_semestre_(aprobadas)"] > 0) & (df["Cantidad_de_evaluaciones_en_el_1er_semestre"] == 0)).sum())),
    ("aprobadas S2 > 0 con evaluaciones S2 = 0", int(((df["Unidades_curriculares_2do_semestre_(aprobadas)"] > 0) & (df["Cantidad_de_evaluaciones_en_el_2do_semestre_"] == 0)).sum())),
    ("nota S1 > 0 con evaluaciones S1 = 0", int(((df["Nota_promedio_en_el_1er_semestre"] > 0) & (df["Cantidad_de_evaluaciones_en_el_1er_semestre"] == 0)).sum())),
    ("nota S2 > 0 con evaluaciones S2 = 0", int(((df["Nota_promedio_en_el_2do_semestre"] > 0) & (df["Cantidad_de_evaluaciones_en_el_2do_semestre_"] == 0)).sum())),
]
pd.DataFrame(checks, columns=["chequeo", "n"]).to_csv(TAB / "04_chequeos_coherencia.csv", index=False)

print(resumen.to_string(index=False))
print("\nTARGET")
print(target.to_string(index=False))
print("\nMOMENTO DE DISPONIBILIDAD")
print(dic_obs["momento"].value_counts().to_string())
print("\nCHEQUEOS")
print(pd.DataFrame(checks, columns=["chequeo","n"]).to_string(index=False))
