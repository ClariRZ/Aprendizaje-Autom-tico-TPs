"""Sensibilidad en train: mismos hiperparámetros y folds entre etapas.

Se fija C=0.1 (Logística) y profundidad=8, hoja=50 (Árbol). Esta comprobación
se hace solamente con CV de train y no reabre ni reevalúa el test.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from comparar_etapas import DATA, OUTPUT, SEED, etapas, preprocesador


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    x = df.drop(columns="target")
    ix_tr, _ = train_test_split(np.arange(len(df)), stratify=df.target,
                                test_size=0.2, random_state=SEED)
    xtr, ytr_multi = x.iloc[ix_tr], df.target.iloc[ix_tr]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    for tarea in ("binaria", "multiclase"):
        y = (ytr_multi == "Desertor").astype(int) if tarea == "binaria" else ytr_multi
        criterio = "roc_auc" if tarea == "binaria" else "f1_macro"
        for etapa, cols in etapas(x.columns.tolist()).items():
            if tarea == "binaria":
                algoritmo = LogisticRegression(C=0.1, max_iter=2000, random_state=SEED)
            else:
                algoritmo = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                                   random_state=SEED)
            modelo = Pipeline([
                ("pre", preprocesador(xtr[cols], escalar=tarea == "binaria")),
                ("modelo", algoritmo),
            ])
            scores = cross_validate(modelo, xtr[cols], y, cv=cv,
                                    scoring=criterio, n_jobs=2, error_score="raise")
            for fold, score in enumerate(scores["test_score"], start=1):
                rows.append({"tarea": tarea, "etapa": etapa, "fold": fold,
                             "criterio": criterio, "score_validacion": score,
                             "config_fija": "C=0.1" if tarea == "binaria" else "depth=8, leaf=50"})
            print(tarea, etapa, np.mean(scores["test_score"]), flush=True)
    pd.DataFrame(rows).to_csv(args.out / "validacion_config_fija.csv", index=False)


if __name__ == "__main__":
    main()
