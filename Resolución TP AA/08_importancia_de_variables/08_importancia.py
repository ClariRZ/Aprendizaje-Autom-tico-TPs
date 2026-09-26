from pathlib import Path
import os
import sys
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from _comun import *

# ---------------------------------------------------------------------------
# Datos
#
# Mantenemos exactamente el split 80/20 usado en los bloques anteriores.
# El test NO se usa para calcular importancia.
# ---------------------------------------------------------------------------
df = cargar()
X, y = xy(df, False)

Xtr, Xtest, ytr, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEMILLA,
)

# ---------------------------------------------------------------------------
# Modelo cuya importancia interpretamos
#
# Es exactamente la configuración de Random Forest seleccionada en bloque 07.
# ---------------------------------------------------------------------------
def crear_modelo():
    return Pipeline([
        ("pre", pre(Xtr, False)),
        ("modelo", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=SEMILLA,
            n_jobs=1,
        )),
    ])

# ---------------------------------------------------------------------------
# Importancia por permutación con validación cruzada dentro de TRAIN.
#
# Cada fold:
#   1) ajusta el RF con 2/3 de train;
#   2) calcula F1 macro en el fold retenido;
#   3) permuta cada variable original 5 veces.
#
# Esto evita usar test para interpretar variables.
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)
N_JOBS = max(1, min(4, os.cpu_count() or 1))
N_REPEATS = 5

fold_rows = []
baseline_rows = []

print("Train para interpretación:", Xtr.shape)
print("Test reservado y NO utilizado:", Xtest.shape)
print("Modelo: Random Forest seleccionado en bloque 07")
print("  n_estimators=500")
print("  max_depth=None")
print("  min_samples_leaf=1")
print("  max_features='sqrt'")
print(f"Permutation importance: 3 folds × {N_REPEATS} repeticiones por variable")
print(f"Paralelismo de permutación: {N_JOBS} procesos")

t0 = time.perf_counter()

for fold, (idx_fit, idx_val) in enumerate(cv.split(Xtr, ytr), start=1):
    print("\n" + "=" * 78)
    print(f"Fold {fold}/3")
    print("=" * 78)

    Xfit = Xtr.iloc[idx_fit]
    Xval = Xtr.iloc[idx_val]
    yfit = ytr.iloc[idx_fit]
    yval = ytr.iloc[idx_val]

    modelo = crear_modelo()
    modelo.fit(Xfit, yfit)

    pred = modelo.predict(Xval)
    baseline = f1_score(yval, pred, average="macro")
    baseline_rows.append({
        "fold": fold,
        "f1_macro_baseline": baseline,
        "n_fit": len(Xfit),
        "n_val": len(Xval),
    })

    print(f"F1 macro baseline: {baseline:.4f}")

    p = permutation_importance(
        modelo,
        Xval,
        yval,
        scoring="f1_macro",
        n_repeats=N_REPEATS,
        random_state=SEMILLA + fold,
        n_jobs=N_JOBS,
    )

    for j, variable in enumerate(X.columns):
        for repeticion, valor in enumerate(p.importances[j], start=1):
            fold_rows.append({
                "fold": fold,
                "variable": variable,
                "repeticion": repeticion,
                "caida_f1_macro": valor,
            })

elapsed = (time.perf_counter() - t0) / 60

detalle = pd.DataFrame(fold_rows)

# ---------------------------------------------------------------------------
# Etiquetas interpretativas derivadas del EDA
# ---------------------------------------------------------------------------
admin_econ = {
    "Pago_de_matrícula_al_día",
    "Tiene_deuda",
    "Poseedor_de_Beca",
}

ingreso_clave = {
    "Edad_al_momento_de_la_inscripción",
    "Puntaje_en_examen_de_ingreso",
    "Cualificación_promedio_de_estudios_previos",
}

def bloque(variable):
    if "1er_semestre" in variable or "2do_semestre" in variable:
        return "Académico"
    if variable in admin_econ:
        return "Administrativo/económico"
    return "Ingreso/contexto"

def hipotesis(variable):
    # Etiquetas del informe final: H1 académico; H2 administrativo/ingreso.
    # H3 (confusión de En Curso) se evalúa con métricas por clase y matriz.
    if "1er_semestre" in variable or "2do_semestre" in variable:
        return "H1"
    if variable in admin_econ or variable in ingreso_clave:
        return "H2"
    return "Contextual/exploratoria"

resumen = (
    detalle.groupby("variable")["caida_f1_macro"]
    .agg(importancia_media="mean", sd="std", mediana="median", minimo="min", maximo="max")
    .reset_index()
)

resumen["bloque"] = resumen["variable"].map(bloque)
resumen["hipotesis_informe"] = resumen["variable"].map(hipotesis)
resumen["importancia_positiva"] = resumen["importancia_media"].clip(lower=0)
resumen = resumen.sort_values("importancia_media", ascending=False).reset_index(drop=True)
resumen["ranking"] = np.arange(1, len(resumen) + 1)

bloques = (
    resumen.groupby("bloque")
    .agg(
        n_variables=("variable", "size"),
        suma_importancia_positiva=("importancia_positiva", "sum"),
        importancia_media_variables=("importancia_media", "mean"),
        max_importancia_variable=("importancia_media", "max"),
    )
    .reset_index()
    .sort_values("suma_importancia_positiva", ascending=False)
)

baselines = pd.DataFrame(baseline_rows)

resumen.to_csv(OUT / "08_importancia.csv", index=False)
detalle.to_csv(OUT / "08_importancia_repeticiones.csv", index=False)
bloques.to_csv(OUT / "08_importancia_bloques.csv", index=False)
baselines.to_csv(OUT / "08_importancia_baseline_cv.csv", index=False)

print("\n" + "=" * 78)
print("TOP 15 — PERMUTATION IMPORTANCE")
print("=" * 78)
print(
    resumen[
        ["ranking", "variable", "bloque", "hipotesis_informe", "importancia_media", "sd"]
    ].head(15).to_string(index=False)
)

print("\n" + "=" * 78)
print("RESUMEN POR BLOQUE")
print("=" * 78)
print(bloques.to_string(index=False))

print(f"\nTiempo total: {elapsed:.2f} min")
print("\nGuardado:")
for pth in [
    OUT / "08_importancia.csv",
    OUT / "08_importancia_repeticiones.csv",
    OUT / "08_importancia_bloques.csv",
    OUT / "08_importancia_baseline_cv.csv",
]:
    print(pth)

print("\nIMPORTANTE:")
print("El test no se utilizó para calcular importancia.")
print("El ranking es predictivo, no causal.")
print("Variables correlacionadas pueden repartirse importancia.")
