from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from _comun import *

# ---------------------------------------------------------------------------
# Misma tarea y mismo split que en el bloque 06
# ---------------------------------------------------------------------------
df = cargar()
X, y = xy(df, False)

Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEMILLA,
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)
N_JOBS = max(1, min(4, os.cpu_count() or 1))

# ---------------------------------------------------------------------------
# Grillas deliberadamente acotadas.
#
# Queremos comparar familias de ensambles y controlar complejidad,
# no realizar una optimización exhaustiva post-hoc.
# ---------------------------------------------------------------------------
rf = Pipeline([
    ("pre", pre(Xtr, False)),
    ("modelo", RandomForestClassifier(
        random_state=SEMILLA,
        n_jobs=1,  # GridSearch paraleliza candidatos; evitamos paralelismo anidado.
    )),
])

rf_grid = {
    "modelo__n_estimators": [200, 500],
    "modelo__max_depth": [12, None],
    "modelo__min_samples_leaf": [1, 50],
    "modelo__max_features": ["sqrt"],
}

ada = Pipeline([
    ("pre", pre(Xtr, False)),
    ("modelo", AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=SEMILLA),
        random_state=SEMILLA,
    )),
])

ada_grid = {
    "modelo__estimator__max_depth": [1, 2],
    "modelo__n_estimators": [100, 200, 400],
    "modelo__learning_rate": [0.05, 0.1],
}

spec = {
    "RandomForest": (rf, rf_grid),
    "AdaBoost": (ada, ada_grid),
}

print("Train:", Xtr.shape, "| Test:", Xte.shape)
print("\nDistribución de clases — train:")
print(ytr.value_counts(normalize=True).sort_index().round(4).to_string())
print("\nDistribución de clases — test:")
print(yte.value_counts(normalize=True).sort_index().round(4).to_string())
print(f"\nParalelismo GridSearch: {N_JOBS} procesos")

rows = []
cv_rows = []
class_rows = []
conf_rows = []

for nombre, (modelo, grilla) in spec.items():
    n_candidatos = int(np.prod([len(v) for v in grilla.values()]))
    n_fits = n_candidatos * cv.get_n_splits()

    print("\n" + "=" * 78)
    print(nombre)
    print("=" * 78)
    print(f"Candidatos: {n_candidatos} | folds: 3 | fits CV: {n_fits}")
    print(f"Grilla: {grilla}")

    search = GridSearchCV(
        estimator=modelo,
        param_grid=grilla,
        scoring="f1_macro",
        cv=cv,
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
        verbose=1,
        error_score="raise",
        return_train_score=False,
    )

    t0 = time.perf_counter()
    search.fit(Xtr, ytr)
    elapsed = (time.perf_counter() - t0) / 60

    pred = search.predict(Xte)
    proba = search.predict_proba(Xte)
    clases = search.best_estimator_.classes_

    auc_macro_ovr = roc_auc_score(
        yte,
        proba,
        multi_class="ovr",
        average="macro",
        labels=clases,
    )

    result = {
        "modelo": nombre,
        "accuracy": accuracy_score(yte, pred),
        "precision_macro": precision_score(yte, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(yte, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(yte, pred, average="macro"),
        "roc_auc_macro_ovr": auc_macro_ovr,
        "cv_f1_macro": search.best_score_,
        "tiempo_min": elapsed,
        "best": json.dumps(search.best_params_, ensure_ascii=False),
    }
    rows.append(result)

    print(f"Mejor CV F1 macro: {search.best_score_:.4f}")
    print(f"Mejores hiperparámetros: {search.best_params_}")
    print(f"Tiempo: {elapsed:.2f} min")

    print("\nResultados en test:")
    print(pd.DataFrame([result]).to_string(index=False))

    print("\nReporte por clase:")
    report = classification_report(
        yte,
        pred,
        output_dict=True,
        zero_division=0,
    )
    print(classification_report(yte, pred, digits=4, zero_division=0))

    for clase in clases:
        r = report[str(clase)]
        class_rows.append({
            "modelo": nombre,
            "clase": clase,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1-score"],
            "support": int(r["support"]),
        })

    cm = confusion_matrix(yte, pred, labels=clases)
    for i, real in enumerate(clases):
        for j, predicha in enumerate(clases):
            conf_rows.append({
                "modelo": nombre,
                "clase_real": real,
                "clase_predicha": predicha,
                "n": int(cm[i, j]),
            })

    res = search.cv_results_
    for params, mean, std, rank in zip(
        res["params"],
        res["mean_test_score"],
        res["std_test_score"],
        res["rank_test_score"],
    ):
        cv_rows.append({
            "modelo": nombre,
            "parametros": json.dumps(params, ensure_ascii=False),
            "cv_f1_macro_media": mean,
            "cv_f1_macro_sd": std,
            "ranking_cv": int(rank),
        })

final = pd.DataFrame(rows)
detalle_cv = pd.DataFrame(cv_rows)
por_clase = pd.DataFrame(class_rows)
confusiones = pd.DataFrame(conf_rows)

# Comparación con el Árbol del bloque 06, si existe.
base06 = OUT / "06_multiclase.csv"
if base06.exists():
    prev = pd.read_csv(base06)
    cols = [
        "modelo", "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "roc_auc_macro_ovr"
    ]
    comparacion = pd.concat(
        [prev[prev["modelo"] == "Arbol"][cols], final[cols]],
        ignore_index=True,
    )
else:
    comparacion = final.copy()

final.to_csv(OUT / "07_ensembles.csv", index=False)
detalle_cv.to_csv(OUT / "07_ensembles_cv.csv", index=False)
por_clase.to_csv(OUT / "07_ensembles_por_clase.csv", index=False)
confusiones.to_csv(OUT / "07_ensembles_confusion.csv", index=False)
comparacion.to_csv(OUT / "07_comparacion_con_arbol.csv", index=False)

print("\n" + "=" * 78)
print("RESUMEN ENSAMBLES EN TEST")
print("=" * 78)
print(final.to_string(index=False))

print("\n" + "=" * 78)
print("COMPARACIÓN CON ÁRBOL MULTICLASE")
print("=" * 78)
print(comparacion.to_string(index=False))

print("\nGuardado:")
for p in [
    OUT / "07_ensembles.csv",
    OUT / "07_ensembles_cv.csv",
    OUT / "07_ensembles_por_clase.csv",
    OUT / "07_ensembles_confusion.csv",
    OUT / "07_comparacion_con_arbol.csv",
]:
    print(p)
