"""Comprobaciones didácticas del TP con el train de la división ya publicada.

Desde la raíz: python "AA privada/03_verificaciones_catedra/experimentos.py" --data 00_base_de_datos/dataset_TP1_2026.csv
No emplea test para seleccionar modelos; todos los tamaños son subconjuntos de train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_alternativa_temporal" / "modelos"))
from comparar_etapas import preprocesador  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "00_base_de_datos" / "dataset_TP1_2026.csv"
OUT = Path(__file__).resolve().parent / "resultados"
SEED = 42
GRAY, RED = "#555555", "#c43c39"


def nested_subset(x, y, n):
    """Misma muestra para todos los modelos de un tamaño; estratificada."""
    if n == len(y):
        return x, y
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=SEED)
    return x.iloc[idx], y.iloc[idx]


def pipeline(x, estimator, scale=False):
    return Pipeline([("pre", preprocesador(x, escalar=scale)), ("modelo", estimator)])


def scores(model, x, y, folds, task, *, proba=False):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for fold, (itr, iva) in enumerate(splitter.split(x, y), 1):
        fitted = clone(model).fit(x.iloc[itr], y.iloc[itr])
        yp_tr = fitted.predict(x.iloc[itr])
        yp_va = fitted.predict(x.iloc[iva])
        if task == "binaria":
            ptr = fitted.predict_proba(x.iloc[itr])[:, list(fitted.classes_).index(1)]
            pva = fitted.predict_proba(x.iloc[iva])[:, list(fitted.classes_).index(1)]
            train, validation = roc_auc_score(y.iloc[itr], ptr), roc_auc_score(y.iloc[iva], pva)
        else:
            train = f1_score(y.iloc[itr], yp_tr, average="macro", zero_division=0)
            validation = f1_score(y.iloc[iva], yp_va, average="macro", zero_division=0)
        row = {"fold": fold, "n_train_fold": len(itr), "n_validation_fold": len(iva),
               "train": train, "validacion": validation}
        if proba:
            probs = fitted.predict_proba(x.iloc[iva])
            for method in ("ovr", "ovo"):
                row[f"auc_macro_{method}"] = roc_auc_score(
                    y.iloc[iva], probs, multi_class=method, average="macro", labels=fitted.classes_)
        rows.append(row)
    return rows


def add(rows, section, model, task, x, y, folds, parameter, value, estimator, *, proba=False):
    print(f"{section}: {model}, {parameter}={value}, {len(y)} filas, {folds} folds", flush=True)
    for r in scores(estimator, x, y, folds, task, proba=proba):
        rows.append({"seccion": section, "modelo": model, "tarea": task,
                     "n_train_disponible": len(y), "folds": folds,
                     "parametro": parameter, "valor": "sin_limite" if value is None else str(value), **r})


def charts(frame, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    for model, part in frame[frame.seccion == "curva_complejidad"].groupby("modelo", sort=False):
        summary = part.groupby("valor", sort=False)[["train", "validacion"]].mean()
        fig, ax = plt.subplots(figsize=(8, 4.4))
        xs = range(len(summary))
        ax.plot(xs, summary.train, marker="o", color=GRAY, lw=2, label="Entrenamiento")
        ax.plot(xs, summary.validacion, marker="o", color=RED, lw=2, label="Validación cruzada")
        ax.set_xticks(list(xs), summary.index)
        ax.set_ylim(max(0, min(summary.min()) - .09), min(1.02, max(summary.max()) + .07))
        ax.set_ylabel("ROC-AUC" if part.tarea.iloc[0] == "binaria" else "F1 macro")
        ax.set_xlabel(part.parametro.iloc[0].replace("_", " "))
        ax.set_title(f"{model}: complejidad y generalización")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="y", alpha=.18)
        fig.text(.12, .01, f"Train disponible: {part.n_train_disponible.iloc[0]:,}; {part.folds.iloc[0]} folds. Eje vertical recortado; comparar valores.", color=GRAY, fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.savefig(out / f"curva_{model.lower()}.svg", bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    if df.shape != (50000, 28) or df.isna().any().any():
        raise ValueError("Se esperaba la base de 50.000 filas, 27 predictores y sin NA")
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=.2,
                                     stratify=df.target, random_state=SEED)
    x = df.drop(columns="target").iloc[idx_tr]
    y = df.target.iloc[idx_tr]
    ybin = (y == "Desertor").astype(int)
    xs, ys = nested_subset(x, y, 12000)
    xk, yk = nested_subset(x, y, 6000)
    rows = []

    # Una sola dimensión cambia por serie: las otras quedan fijas.
    for depth in (4, 8, 12, None):
        add(rows, "curva_complejidad", "Arbol", "multiclase", x, y, 3,
            "max_depth", depth, pipeline(x, DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=50, random_state=SEED)))
    for depth in (4, 8, 12, None):
        add(rows, "curva_complejidad", "RandomForest", "multiclase", xs, ys, 3,
            "max_depth", depth, pipeline(xs, RandomForestClassifier(
                n_estimators=80, max_depth=depth, min_samples_leaf=10,
                max_features="sqrt", bootstrap=True, random_state=SEED, n_jobs=2)))
    for trees in (20, 60, 120):
        add(rows, "curva_complejidad", "AdaBoost", "multiclase", xs, ys, 3,
            "n_estimators", trees, pipeline(xs, AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
                n_estimators=trees, learning_rate=.1, random_state=SEED)))
    for k in (3, 11, 31, 101):
        add(rows, "curva_complejidad", "KNN", "multiclase", xk, yk, 3,
            "n_neighbors", k, pipeline(xk, KNeighborsClassifier(n_neighbors=k), scale=True))

    # Segunda pasada pequeña: dos parámetros simultáneos, sin tocar el test.
    for depth in (6, 8, 12, None):
        for leaf in (10, 50, 100):
            add(rows, "busqueda_2_parametros", "Arbol", "multiclase", x, y, 3,
                "max_depth,min_samples_leaf", f"{depth},{leaf}",
                pipeline(x, DecisionTreeClassifier(
                    max_depth=depth, min_samples_leaf=leaf, random_state=SEED)))

    # Modelos fijos: comparar 3 y 5 folds no implica una búsqueda ni escoger el mejor número.
    fixed = [
        ("Dummy_binario", "binaria", x, ybin, DummyClassifier(strategy="most_frequent")),
        ("Logistica", "binaria", x, ybin, pipeline(x, LogisticRegression(
            C=.1, max_iter=2000, random_state=SEED), scale=True)),
        ("Dummy_multiclase", "multiclase", x, y, DummyClassifier(strategy="most_frequent")),
        ("Arbol", "multiclase", x, y, pipeline(x, DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=50, random_state=SEED))),
        ("RandomForest", "multiclase", xs, ys, pipeline(xs, RandomForestClassifier(
            n_estimators=80, max_depth=12, min_samples_leaf=10,
            max_features="sqrt", bootstrap=True, random_state=SEED, n_jobs=2))),
    ]
    for folds in (3, 5):
        for name, task, xx, yy, estimator in fixed:
            add(rows, "sensibilidad_folds", name, task, xx, yy, folds, "n_splits", folds,
                estimator, proba=task == "multiclase" and name in ("Arbol", "RandomForest"))

    # Curva de tamaño con árbol fijo, dentro del train y siempre con 3 folds.
    for n in (10000, 20000, 40000):
        xx, yy = nested_subset(x, y, n)
        add(rows, "tamano_train", "Arbol", "multiclase", xx, yy, 3,
            "n_train_disponible", n, pipeline(xx, DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=50, random_state=SEED)))

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "scores_folds.csv", index=False, float_format="%.9f")
    charts(frame, args.out)
    info = {"dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "filas_totales": len(df), "train": len(idx_tr), "test_no_usado": len(idx_te),
            "split": "estratificado por tres clases; random_state=42; test_size=0.20",
            "folds": "StratifiedKFold con shuffle=True y random_state=42; preprocesamiento ajustado dentro de cada fold",
            "scores": "ROC-AUC binaria / F1 macro multiclase; OvR y OvO sobre mismas probabilidades por fold",
            "cuidado": "la base se analizó antes; sensibilidad interna, no validación externa"}
    (args.out / "protocolo.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Resultados guardados:", args.out, flush=True)


if __name__ == "__main__":
    main()
