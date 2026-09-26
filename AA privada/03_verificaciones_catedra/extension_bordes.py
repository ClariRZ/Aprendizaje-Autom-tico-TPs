"""Sensibilidad de K-NN y AdaBoost a valores fuera de la grilla inicial.

Desde la raíz del repositorio:
  python "AA privada/03_verificaciones_catedra/extension_bordes.py" \
    --data "00_base_de_datos/dataset_TP1_2026.csv"

Sólo utiliza el 80 % de entrenamiento estratificado por tres clases. No vuelve
a escoger un modelo ni calcula métricas sobre test. Para AdaBoost aprovecha
staged_predict: las predicciones tras 100, 200, 400 y 600 árboles provienen
del mismo ajuste y se calculan sobre los mismos folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "02_alternativa_temporal" / "modelos"))
from comparar_etapas import preprocesador  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "00_base_de_datos" / "dataset_TP1_2026.csv"
OUT = Path(__file__).resolve().parent / "resultados"
SEED = 42


def f1(y, pred):
    return f1_score(y, pred, average="macro", zero_division=0)


def split_folds(x, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (fit, val) in enumerate(cv.split(x, y), 1):
        yield fold, x.iloc[fit], x.iloc[val], y.iloc[fit], y.iloc[val]


def knn_cv(x, y):
    """Un solo preprocesamiento y una consulta de vecinos por fold."""
    values = (25, 35, 51)
    rows = []
    for fold, xf, xv, yf, yv in split_folds(x, y):
        p = preprocesador(xf, escalar=True)
        a, b = p.fit_transform(xf), p.transform(xv)
        m = KNeighborsClassifier(n_neighbors=max(values), weights="uniform",
                                 n_jobs=2).fit(a, yf)
        indexes = m.kneighbors(b, n_neighbors=max(values), return_distance=False)
        class_names = m.classes_
        fit_labels = np.searchsorted(class_names, yf.to_numpy())
        neighbors = fit_labels[indexes]
        for k in values:
            counts = np.stack([(neighbors[:, :k] == cl).sum(axis=1)
                               for cl in range(len(class_names))], axis=1)
            pred = class_names[counts.argmax(axis=1)]
            if k == 25:
                m.set_params(n_neighbors=k)
                assert np.array_equal(pred[:100], m.predict(b[:100]))
                m.set_params(n_neighbors=max(values))
            rows.append({"modelo": "K-NN", "muestra": "train_completo",
                         "filas_disponibles": len(y), "fold": fold,
                         "parametro": "n_neighbors", "valor": k,
                         "otros_parametros": "weights=uniform",
                         "n_ajuste_fold": len(yf), "n_validacion_fold": len(yv),
                         "f1_macro_validacion": f1(yv, pred)})
            print("K-NN", fold, k, rows[-1]["f1_macro_validacion"], flush=True)
    return rows


def ada_cv(x, y, values, *, curve):
    """Profundidad 2 y tasa 0,1: variar exclusivamente número de árboles."""
    rows = []
    for fold, xf, xv, yf, yv in split_folds(x, y):
        model = Pipeline([
            ("pre", preprocesador(xf, escalar=False)),
            ("modelo", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED),
                n_estimators=max(values), learning_rate=.1, random_state=SEED)),
        ]).fit(xf, yf)
        # El mismo preprocesamiento del fold se aplica a ambos subconjuntos.
        a = model.named_steps["pre"].transform(xf) if curve else None
        b = model.named_steps["pre"].transform(xv)
        estimator = model.named_steps["modelo"]
        val_predictions = estimator.staged_predict(b)
        fit_predictions = estimator.staged_predict(a) if curve else None
        for n in range(1, max(values)+1):
            pred_v = next(val_predictions)
            pred_f = next(fit_predictions) if curve else None
            if n not in values:
                continue
            r = {"modelo": "AdaBoost", "muestra": "train_12000" if curve else "train_completo",
                 "filas_disponibles": len(y), "fold": fold,
                 "parametro": "n_estimators", "valor": n,
                 "otros_parametros": "max_depth=2; learning_rate=0.1",
                 "n_ajuste_fold": len(yf), "n_validacion_fold": len(yv),
                 "f1_macro_validacion": f1(yv, pred_v)}
            if curve:
                r["f1_macro_ajuste"] = f1(yf, pred_f)
            rows.append(r)
            print("AdaBoost", "curva" if curve else "train completo",
                  fold, n, r["f1_macro_validacion"], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--section", choices=["all", "knn", "ada_curve", "ada_full"],
                        default="all")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(args.data)
    if df.shape != (50000, 28) or df.isna().any().any():
        raise ValueError("Base distinta de 50.000 filas, 27 predictores sin NA")
    idx_train, _ = train_test_split(np.arange(len(df)), test_size=.20,
                                    stratify=df["target"], random_state=SEED)
    x, y = df.drop(columns="target").iloc[idx_train], df["target"].iloc[idx_train]
    if args.section in ("all", "knn"):
        pd.DataFrame(knn_cv(x, y)).to_csv(args.out / "extension_knn_cv.csv", index=False)
    if args.section in ("all", "ada_curve"):
        indices, _ = train_test_split(np.arange(len(y)), train_size=12000,
                                      stratify=y, random_state=SEED)
        subset_x, subset_y = x.iloc[indices], y.iloc[indices]
        pd.DataFrame(ada_cv(subset_x, subset_y, (100, 200, 400, 600), curve=True)).to_csv(
            args.out / "extension_ada_curva.csv", index=False)
    if args.section in ("all", "ada_full"):
        pd.DataFrame(ada_cv(x, y, (200, 400, 600), curve=False)).to_csv(
            args.out / "extension_ada_cv.csv", index=False)
    info = {"dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "sklearn_version": sklearn.__version__,
            "split": "80/20, stratify=target, random_state=42; test sin utilizar",
            "cv": "StratifiedKFold(3, shuffle=True, random_state=42)",
            "seleccion": "sensibilidad posterior a la grilla original; no se actualiza el test",
            "nota": "si la validación sigue subiendo en un extremo, no es evidencia de óptimo"}
    (args.out / "extension_bordes_protocolo.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
