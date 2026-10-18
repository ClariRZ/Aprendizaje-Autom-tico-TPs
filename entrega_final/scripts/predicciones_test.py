"""Reajusta los ocho modelos finales y guarda sus puntajes en test.

Desde la raíz del repositorio:
    python 00_base_de_datos/reconstruir_dataset.py
    python entrega_final/scripts/predicciones_test.py

Usa las configuraciones de la Tabla 1, las mismas particiones (semilla 42)
y el preprocesamiento de `Resolución TP AA/_comun`. Comprueba que las
métricas coincidan con `verificaciones/metricas_reproducidas.csv` al
redondear a cuatro decimales y guarda en `datos_figuras/`:
- puntajes_binaria.csv: probabilidad o función de decisión por modelo;
- predicciones_multiclase.csv: clase predicha por modelo.
No selecciona nada con el test: solo registra salidas para las figuras.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Resolución TP AA"))
from _comun import SEMILLA, cargar, pre, xy  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "datos_figuras"
OUT.mkdir(exist_ok=True)
REF = pd.read_csv(Path(__file__).resolve().parents[1] / "verificaciones" / "metricas_reproducidas.csv")

df = cargar()

# --- Binaria ---------------------------------------------------------------
X, y = xy(df, True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
lr = Pipeline([("pre", pre(Xtr, True)), ("modelo", LogisticRegression(C=0.1, max_iter=3000))])
svm = Pipeline([("pre", pre(Xtr, True)),
                ("modelo", SVC(C=0.1, gamma=0.01, cache_size=500, random_state=SEMILLA))])
svm_cal = CalibratedClassifierCV(
    estimator=Pipeline([("pre", pre(Xtr, True)),
                        ("modelo", SVC(C=0.1, gamma=0.01, cache_size=500, random_state=SEMILLA))]),
    method="sigmoid", cv=3, ensemble=False)
tree = Pipeline([("pre", pre(Xtr, False)),
                 ("modelo", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=SEMILLA))])
voting = VotingClassifier([("logistica", lr), ("svm", svm_cal), ("arbol", tree)], voting="soft")

puntajes = pd.DataFrame({"y": yte.to_numpy()})
for nombre, modelo in [("Logistica", lr), ("Arbol", tree), ("Soft Voting", voting)]:
    modelo.fit(Xtr, ytr)
    puntajes[nombre] = modelo.predict_proba(Xte)[:, 1]
    print(nombre, "listo", flush=True)
svm.fit(Xtr, ytr)
puntajes["SVM"] = svm.decision_function(Xte)
puntajes.to_csv(OUT / "puntajes_binaria.csv", index=False)

for nombre in ["Logistica", "SVM", "Arbol", "Soft Voting"]:
    s = puntajes[nombre]
    pred = (s > 0).astype(int) if nombre == "SVM" else (s >= 0.5).astype(int)
    ref = REF[(REF.tarea == "binaria") & (REF.modelo == nombre)].iloc[0]
    got = (round(roc_auc_score(yte, s), 4), round(f1_score(yte, pred), 4))
    exp = (round(ref.roc_auc, 4), round(ref.f1, 4))
    print(f"{nombre}: AUC, F1 = {got} | referencia {exp}", "OK" if got == exp else "DIFERENCIA")

# --- Multiclase ------------------------------------------------------------
X, y = xy(df, False)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
modelos = {
    "KNN": Pipeline([("pre", pre(Xtr, True)),
                     ("modelo", KNeighborsClassifier(n_neighbors=25, weights="uniform"))]),
    "Arbol_multiclase": Pipeline([("pre", pre(Xtr, False)),
                                  ("modelo", DecisionTreeClassifier(max_depth=12, min_samples_leaf=50,
                                                                    random_state=SEMILLA))]),
    "RandomForest": Pipeline([("pre", pre(Xtr, False)),
                              ("modelo", RandomForestClassifier(n_estimators=500, max_depth=None,
                                                                min_samples_leaf=1, max_features="sqrt",
                                                                random_state=SEMILLA, n_jobs=-1))]),
    "AdaBoost": Pipeline([("pre", pre(Xtr, False)),
                          ("modelo", AdaBoostClassifier(
                              estimator=DecisionTreeClassifier(max_depth=2, random_state=SEMILLA),
                              n_estimators=400, learning_rate=0.1, random_state=SEMILLA))]),
}
pred = pd.DataFrame({"y": yte.to_numpy()})
for nombre, modelo in modelos.items():
    modelo.fit(Xtr, ytr)
    pred[nombre] = modelo.predict(Xte)
    ref = REF[(REF.tarea == "multiclase") & (REF.modelo == nombre)].iloc[0]
    got = round(f1_score(yte, pred[nombre], average="macro"), 4)
    print(f"{nombre}: F1 macro = {got} | referencia {round(ref.f1_macro, 4)}",
          "OK" if got == round(ref.f1_macro, 4) else "DIFERENCIA", flush=True)
pred.to_csv(OUT / "predicciones_multiclase.csv", index=False)
print("Guardado en", OUT)
