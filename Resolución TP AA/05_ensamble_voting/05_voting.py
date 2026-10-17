from pathlib import Path
import os
import sys
import time

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from _comun import *

# ---------------------------------------------------------------------------
# Datos: mismo split que en el bloque 03
# ---------------------------------------------------------------------------
df = cargar()
X, y = xy(df, True)

Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEMILLA,
)

N_JOBS = max(1, min(4, os.cpu_count() or 1))

# ---------------------------------------------------------------------------
# Modelos base: exactamente los hiperparámetros seleccionados en bloque 03
# ---------------------------------------------------------------------------
lr = Pipeline([
    ("pre", pre(Xtr, True)),
    ("modelo", LogisticRegression(
        C=0.1,
        max_iter=3000,
    )),
])

# La calibración divide el entrenamiento en folds internos. El estimador
# completo (preprocesamiento + SVM) debe reajustarse dentro de cada fold:
# el escalador y los imputadores tampoco deben ver la parte de calibración.
svm = CalibratedClassifierCV(
    estimator=Pipeline([
        ("pre", pre(Xtr, True)),
        ("modelo", SVC(
            C=0.1,
            gamma=0.01,
            cache_size=500,
            random_state=SEMILLA,
        )),
    ]),
    method="sigmoid",
    cv=3,
    ensemble=False,
)

tree = Pipeline([
    ("pre", pre(Xtr, False)),
    ("modelo", DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=50,
        random_state=SEMILLA,
    )),
])

# ---------------------------------------------------------------------------
# Soft Voting
#
# Pesos iguales decididos a priori.
# No se optimizan pesos con test.
# ---------------------------------------------------------------------------
voting = VotingClassifier(
    estimators=[
        ("logistica", lr),
        ("svm", svm),
        ("arbol", tree),
    ],
    voting="soft",
    weights=None,
    n_jobs=N_JOBS,
)

print("Train:", Xtr.shape, "| Test:", Xte.shape)
print(f"Prevalencia Desertor — train: {ytr.mean():.4f} | test: {yte.mean():.4f}")
print("\nModelos base:")
print("  Logística: C=0.1")
print("  SVM RBF:   C=0.1, gamma=0.01 + calibración sigmoide")
print("  Árbol:     max_depth=8, min_samples_leaf=50")
print("Voting: soft, pesos iguales")

t0 = time.perf_counter()
voting.fit(Xtr, ytr)
elapsed = (time.perf_counter() - t0) / 60

pred = voting.predict(Xte)
prob = voting.predict_proba(Xte)[:, 1]

tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()

resultado = pd.DataFrame([{
    "modelo": "Voting soft",
    "accuracy": accuracy_score(yte, pred),
    "precision": precision_score(yte, pred),
    "recall": recall_score(yte, pred),
    "f1": f1_score(yte, pred),
    "roc_auc": roc_auc_score(yte, prob),
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp,
    "tiempo_min": elapsed,
}])

# Comparación directa con bloque 03 si existe
base_path = OUT / "03_binario.csv"
if base_path.exists():
    base = pd.read_csv(base_path)
    cols = ["modelo", "accuracy", "precision", "recall", "f1", "roc_auc"]
    comparacion = pd.concat(
        [base[cols], resultado[cols]],
        ignore_index=True,
    )
else:
    comparacion = resultado.copy()

resultado.to_csv(OUT / "05_voting.csv", index=False)
comparacion.to_csv(OUT / "05_comparacion_voting.csv", index=False)

print("\n" + "=" * 72)
print("RESULTADO VOTING EN TEST")
print("=" * 72)
print(resultado.to_string(index=False))

print("\n" + "=" * 72)
print("COMPARACIÓN CON MODELOS BASE")
print("=" * 72)
print(comparacion.to_string(index=False))

print("\nGuardado:")
print(OUT / "05_voting.csv")
print(OUT / "05_comparacion_voting.csv")
