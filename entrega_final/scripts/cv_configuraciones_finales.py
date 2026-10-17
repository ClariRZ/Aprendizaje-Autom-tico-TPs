"""Validación cruzada de las configuraciones finales (media y DE entre folds).

Desde la raíz del repositorio:
    python entrega_final/scripts/cv_configuraciones_finales.py

Usa la misma validación de la búsqueda de hiperparámetros: StratifiedKFold de
tres folds, shuffle=True, semilla 42, dentro de los 40.000 registros de
entrenamiento. Solo describe las configuraciones ya elegidas (Tabla 2); no
selecciona nada. El Soft Voting no se eligió por validación cruzada y no se
incluye. La DE usa ddof=0 y no es un intervalo de confianza.
"""
from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Resolución TP AA"))
from _comun import SEMILLA, cargar, pre, xy  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "datos_figuras"
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)
df = cargar()
filas = []


def evaluar(tarea, nombre, modelo, X, y, metrica):
    s = cross_val_score(modelo, X, y, cv=cv, scoring=metrica, n_jobs=1)
    filas.append({"tarea": tarea, "modelo": nombre, "metrica": metrica,
                  "media": s.mean(), "de": s.std(ddof=0)})
    print(f"{tarea:10s} {nombre:18s} {metrica}: {s.mean():.4f} ± {s.std(ddof=0):.4f}", flush=True)


X, y = xy(df, True)
Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
evaluar("binaria", "Logistica", Pipeline([("pre", pre(Xtr, True)),
        ("modelo", LogisticRegression(C=0.1, max_iter=3000))]), Xtr, ytr, "roc_auc")
evaluar("binaria", "Arbol", Pipeline([("pre", pre(Xtr, False)),
        ("modelo", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=SEMILLA))]),
        Xtr, ytr, "roc_auc")
evaluar("binaria", "SVM", Pipeline([("pre", pre(Xtr, True)),
        ("modelo", SVC(C=0.1, gamma=0.01, cache_size=500))]), Xtr, ytr, "roc_auc")

X, y = xy(df, False)
Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
mc = {
    "KNN": Pipeline([("pre", pre(Xtr, True)), ("modelo", KNeighborsClassifier(n_neighbors=25))]),
    "Arbol_multiclase": Pipeline([("pre", pre(Xtr, False)), ("modelo", DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=50, random_state=SEMILLA))]),
    "RandomForest": Pipeline([("pre", pre(Xtr, False)), ("modelo", RandomForestClassifier(
        n_estimators=500, max_features="sqrt", random_state=SEMILLA, n_jobs=-1))]),
    "AdaBoost": Pipeline([("pre", pre(Xtr, False)), ("modelo", AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=SEMILLA),
        n_estimators=400, learning_rate=0.1, random_state=SEMILLA))]),
}
for nombre, modelo in mc.items():
    evaluar("multiclase", nombre, modelo, Xtr, ytr, "f1_macro")

pd.DataFrame(filas).to_csv(OUT / "cv_configuraciones_finales.csv", index=False)
print("Guardado")
