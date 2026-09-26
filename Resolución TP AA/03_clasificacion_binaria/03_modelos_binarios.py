from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from _comun import *

# ---------------------------------------------------------------------------
# Datos y partición final
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

# La selección de hiperparámetros ocurre sólo dentro de train.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)

# Evitamos usar todos los núcleos: varias instancias RBF-SVM simultáneas
# pueden competir por RAM/cache y terminar siendo más lentas.
N_JOBS = max(1, min(4, os.cpu_count() or 1))

rows = []
cv_rows = []

print(f"Train: {Xtr.shape} | Test: {Xte.shape}")
print(f"Prevalencia Desertor — train: {ytr.mean():.4f} | test: {yte.mean():.4f}")
print(f"Paralelismo GridSearch: {N_JOBS} procesos")

# Cache temporal: GridSearch prueba muchos hiperparámetros sobre los mismos
# folds. El preprocesamiento idéntico se puede reutilizar sin tocar test.
with TemporaryDirectory(prefix="aa_grid_cache_") as cache_dir:
    memory = Memory(location=cache_dir, verbose=0)

    spec = {
        "Logistica": (
            Pipeline(
                [
                    ("pre", pre(Xtr, True)),
                    ("modelo", LogisticRegression(max_iter=3000)),
                ],
                memory=memory,
            ),
            {"modelo__C": [0.01, 0.1, 1, 10]},
        ),
        "SVM": (
            Pipeline(
                [
                    ("pre", pre(Xtr, True)),
                    # SVC usa kernel RBF por defecto. El cache del kernel ayuda
                    # a evitar recomputaciones costosas.
                    ("modelo", SVC(cache_size=500)),
                ],
                memory=memory,
            ),
            {
                "modelo__C": [0.1, 1, 10],
                "modelo__gamma": ["scale", 0.01, 0.1],
            },
        ),
        "Arbol": (
            Pipeline(
                [
                    ("pre", pre(Xtr, False)),
                    ("modelo", DecisionTreeClassifier(random_state=SEMILLA)),
                ],
                memory=memory,
            ),
            {
                "modelo__max_depth": [2, 4, 6, 8, 12, None],
                "modelo__min_samples_leaf": [1, 10, 50],
            },
        ),
    }

    for nombre, (modelo, grilla) in spec.items():
        n_candidatos = int(np.prod([len(v) for v in grilla.values()]))
        n_fits = n_candidatos * cv.get_n_splits()

        print("\n" + "=" * 72)
        print(nombre)
        print("=" * 72)
        print(f"Candidatos: {n_candidatos} | folds: 3 | fits CV: {n_fits}")
        print(f"Grilla: {grilla}")

        search = GridSearchCV(
            estimator=modelo,
            param_grid=grilla,
            scoring="roc_auc",
            cv=cv,
            n_jobs=N_JOBS,
            pre_dispatch=N_JOBS,
            verbose=1,
            error_score="raise",
            return_train_score=False,
        )

        t0 = time.perf_counter()
        search.fit(Xtr, ytr)
        elapsed = time.perf_counter() - t0

        print(f"Mejor CV ROC-AUC: {search.best_score_:.4f}")
        print(f"Mejores hiperparámetros: {search.best_params_}")
        print(f"Tiempo: {elapsed / 60:.2f} min")

        pred = search.predict(Xte)
        score = (
            search.decision_function(Xte)
            if hasattr(search, "decision_function")
            else search.predict_proba(Xte)[:, 1]
        )

        rows.append(
            {
                "modelo": nombre,
                "accuracy": accuracy_score(yte, pred),
                "precision": precision_score(yte, pred),
                "recall": recall_score(yte, pred),
                "f1": f1_score(yte, pred),
                "roc_auc": roc_auc_score(yte, score),
                "cv_roc_auc": search.best_score_,
                "tiempo_min": elapsed / 60,
                "best": json.dumps(search.best_params_, ensure_ascii=False),
            }
        )

        res = search.cv_results_
        for params, mean, std, rank in zip(
            res["params"],
            res["mean_test_score"],
            res["std_test_score"],
            res["rank_test_score"],
        ):
            cv_rows.append(
                {
                    "modelo": nombre,
                    "parametros": json.dumps(params, ensure_ascii=False),
                    "cv_roc_auc_media": mean,
                    "cv_roc_auc_sd": std,
                    "ranking_cv": rank,
                }
            )

final = pd.DataFrame(rows)
detalle_cv = pd.DataFrame(cv_rows)

final.to_csv(OUT / "03_binario.csv", index=False)
detalle_cv.to_csv(OUT / "03_cv_busqueda.csv", index=False)

print("\n" + "=" * 72)
print("RESULTADOS EN TEST")
print("=" * 72)
print(final.to_string(index=False))

print("\nGuardado:")
print(OUT / "03_binario.csv")
print(OUT / "03_cv_busqueda.csv")
