from pathlib import Path
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from _comun import *

# ---------------------------------------------------------------------------
# Datos: usamos el mismo split 80/20 del bloque 03.
# El test se reserva y NO participa en las curvas de complejidad.
# ---------------------------------------------------------------------------
df = cargar()
X, y = xy(df, True)

Xtr, Xtest, ytr, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEMILLA,
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)
N_JOBS = max(1, min(4, os.cpu_count() or 1))

OUT04 = OUT / "04_complejidad"
FIG = OUT04 / "graficos"
OUT04.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

rows = []

def correr_curva(nombre, modelo, param_name, valores, etiqueta_x, archivo):
    print("\n" + "=" * 78)
    print(nombre)
    print("=" * 78)
    print(f"Parámetro: {param_name}")
    print(f"Valores: {valores}")

    t0 = time.perf_counter()

    train_scores, val_scores = validation_curve(
        modelo,
        Xtr,
        ytr,
        param_name=param_name,
        param_range=valores,
        cv=cv,
        scoring="roc_auc",
        n_jobs=N_JOBS,
    )

    elapsed = (time.perf_counter() - t0) / 60

    train_mean = train_scores.mean(axis=1)
    train_sd = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_sd = val_scores.std(axis=1)

    for v, tm, ts, vm, vs in zip(valores, train_mean, train_sd, val_mean, val_sd):
        rows.append({
            "curva": nombre,
            "parametro": param_name,
            "valor": str(v),
            "auc_train_media": tm,
            "auc_train_sd": ts,
            "auc_val_media": vm,
            "auc_val_sd": vs,
        })

    fig, ax = plt.subplots(figsize=(8, 5))
    xpos = np.arange(len(valores))

    ax.plot(xpos, train_mean, marker="o", label="Train")
    ax.plot(xpos, val_mean, marker="o", label="Validación CV")

    ax.fill_between(
        xpos,
        train_mean - train_sd,
        train_mean + train_sd,
        alpha=0.15,
    )
    ax.fill_between(
        xpos,
        val_mean - val_sd,
        val_mean + val_sd,
        alpha=0.15,
    )

    ax.set_xticks(xpos)
    ax.set_xticklabels([str(v) for v in valores])
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(nombre)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / archivo, dpi=180)
    plt.close(fig)

    print(f"Tiempo: {elapsed:.2f} min")
    print(pd.DataFrame({
        etiqueta_x: valores,
        "auc_train": train_mean,
        "auc_val": val_mean,
        "gap_train_val": train_mean - val_mean,
    }).to_string(index=False))


# ---------------------------------------------------------------------------
# 1. Regresión Logística
# Mejor valor en bloque 03: C = 0.1
# Curva: variar C.
# ---------------------------------------------------------------------------
logit = Pipeline([
    ("pre", pre(Xtr, True)),
    ("modelo", LogisticRegression(max_iter=3000)),
])

correr_curva(
    nombre="Regresión Logística — complejidad según C",
    modelo=logit,
    param_name="modelo__C",
    valores=[0.01, 0.1, 1, 10],
    etiqueta_x="C",
    archivo="01_logistica_C.png",
)


# ---------------------------------------------------------------------------
# 2. SVM RBF
# Mejor bloque 03: C = 0.1, gamma = 0.01.
# Hacemos dos curvas, variando un hiperparámetro a la vez.
# ---------------------------------------------------------------------------
svm_c = Pipeline([
    ("pre", pre(Xtr, True)),
    ("modelo", SVC(gamma=0.01, cache_size=500)),
])

correr_curva(
    nombre="SVM RBF — complejidad según C (gamma=0.01)",
    modelo=svm_c,
    param_name="modelo__C",
    valores=[0.1, 1, 10],
    etiqueta_x="C",
    archivo="02_svm_C.png",
)

svm_gamma = Pipeline([
    ("pre", pre(Xtr, True)),
    ("modelo", SVC(C=0.1, cache_size=500)),
])

correr_curva(
    nombre="SVM RBF — complejidad según gamma (C=0.1)",
    modelo=svm_gamma,
    param_name="modelo__gamma",
    valores=["scale", 0.01, 0.1],
    etiqueta_x="gamma",
    archivo="03_svm_gamma.png",
)


# ---------------------------------------------------------------------------
# 3. Árbol
# Mejor bloque 03: max_depth=8, min_samples_leaf=50.
# Variamos un parámetro y fijamos el otro en su valor seleccionado.
# ---------------------------------------------------------------------------
tree_depth = Pipeline([
    ("pre", pre(Xtr, False)),
    ("modelo", DecisionTreeClassifier(
        min_samples_leaf=50,
        random_state=SEMILLA,
    )),
])

correr_curva(
    nombre="Árbol — complejidad según profundidad (min_leaf=50)",
    modelo=tree_depth,
    param_name="modelo__max_depth",
    valores=[1, 2, 3, 4, 6, 8, 12, 20, None],
    etiqueta_x="max_depth",
    archivo="04_arbol_depth.png",
)

tree_leaf = Pipeline([
    ("pre", pre(Xtr, False)),
    ("modelo", DecisionTreeClassifier(
        max_depth=8,
        random_state=SEMILLA,
    )),
])

correr_curva(
    nombre="Árbol — complejidad según tamaño mínimo de hoja (depth=8)",
    modelo=tree_leaf,
    param_name="modelo__min_samples_leaf",
    valores=[1, 5, 10, 25, 50, 100, 250],
    etiqueta_x="min_samples_leaf",
    archivo="05_arbol_min_leaf.png",
)


detalle = pd.DataFrame(rows)
detalle.to_csv(OUT04 / "04_curvas_complejidad.csv", index=False)

print("\n" + "=" * 78)
print("CURVAS DE COMPLEJIDAD COMPLETAS")
print("=" * 78)
print("El conjunto de test NO fue utilizado.")
print("Resultados:")
print(OUT04 / "04_curvas_complejidad.csv")
print("Gráficos:")
print(FIG)
