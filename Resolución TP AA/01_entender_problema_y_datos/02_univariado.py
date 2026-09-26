from pathlib import Path
import sys
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from _comun import cargar_eda_train, OUT

BASE = OUT / "eda_exploratorio"
TAB = BASE / "tablas"
TAB.mkdir(parents=True, exist_ok=True)

df = cargar_eda_train()
pred = df.drop(columns="target")

num_rows = []
for c in pred.select_dtypes(include=np.number).columns:
    x = pred[c].dropna()
    q25, q75 = x.quantile([.25, .75])
    num_rows.append({
        "variable": c,
        "n": len(x),
        "media": x.mean(),
        "sd": x.std(),
        "mediana": x.median(),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "min": x.min(),
        "max": x.max(),
        "pct_ceros": 100 * x.eq(0).mean(),
        "asimetria": x.skew(),
    })
num = pd.DataFrame(num_rows)
num["flag_muchos_ceros"] = num["pct_ceros"] >= 20
num["flag_asimetria"] = num["asimetria"].abs() >= 1
num.to_csv(TAB / "05_univariado_numericas.csv", index=False)

cat_levels = []
cat_summary = []
for c in pred.select_dtypes(exclude=np.number).columns:
    vc = pred[c].value_counts(dropna=False)
    cat_summary.append({
        "variable": c,
        "n_categorias": int(len(vc)),
        "categoria_mas_frecuente": str(vc.index[0]),
        "pct_categoria_mas_frecuente": 100 * vc.iloc[0] / len(pred),
        "n_categorias_menor_100": int((vc < 100).sum()),
        "min_n_categoria": int(vc.min()),
    })
    for categoria, n in vc.items():
        cat_levels.append({
            "variable": c,
            "categoria": str(categoria),
            "n": int(n),
            "porcentaje": 100 * n / len(pred),
            "flag_n_menor_100": bool(n < 100),
        })

cat_summary = pd.DataFrame(cat_summary)
cat_levels = pd.DataFrame(cat_levels)
cat_summary.to_csv(TAB / "06_resumen_categoricas.csv", index=False)
cat_levels.to_csv(TAB / "07_niveles_categoricas.csv", index=False)

print("NUMÉRICAS")
print(num.to_string(index=False))
print("\nCATEGÓRICAS")
print(cat_summary.to_string(index=False))
