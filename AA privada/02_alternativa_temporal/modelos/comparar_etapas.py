"""Comparación exploratoria por disponibilidad de variables del TP1.

Ejecución desde la raíz del repositorio:
    python "AA privada/02_alternativa_temporal/modelos/comparar_etapas.py"

No selecciona modelos o hipótesis con el test. Este dataset ya se analizó en el
TP original; los resultados del test compartido son exploratorios, no una
confirmación independiente.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


SEED = 42
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "00_base_de_datos" / "dataset_TP1_2026.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "resultados_modelos"

S1 = [
    "Unidades_curriculares_1er_semestre_(inscrito)",
    "Cantidad_de_evaluaciones_en_el_1er_semestre",
    "Unidades_curriculares_1er_semestre_(aprobadas)",
    "Nota_promedio_en_el_1er_semestre",
]
S2 = [
    "Unidades_curriculares_2do_semestre_(inscrito)",
    "Cantidad_de_evaluaciones_en_el_2do_semestre_",
    "Unidades_curriculares_2do_semestre_(aprobadas)",
    "Nota_promedio_en_el_2do_semestre",
]
# Estas siete columnas no tienen fecha de captura verificada en el diccionario.
SIN_FECHA = [
    "Tiene_deuda", "Pago_de_matrícula_al_día", "Poseedor_de_Beca",
    "Demanda_Cog_Padre", "Demanda_Cog_Madre",
    "Ingresos_Padre", "Ingresos_Madre",
]


def etapas(columnas: list[str]) -> dict[str, list[str]]:
    if len(columnas) != 27 or len(set(columnas)) != 27:
        raise ValueError("Se esperaba la base original con 27 predictores únicos")
    if not set(S1 + S2 + SIN_FECHA) <= set(columnas):
        raise ValueError("Faltan columnas esperadas; verificar versión del dataset")
    entrada = [c for c in columnas if c not in S1 + S2 + SIN_FECHA]
    if len(entrada) != 12:
        raise ValueError("Se esperaban 12 variables candidatas de ingreso")
    return {
        "E0_entrada_candidata": entrada,
        "E1_mas_semestre1": entrada + S1,
        "E2_mas_semestre2": entrada + S1 + S2,
        "E0_con_sin_fecha": entrada + SIN_FECHA,
        "E1_con_sin_fecha": entrada + SIN_FECHA + S1,
        "E2_con_sin_fecha": entrada + SIN_FECHA + S1 + S2,
    }


def preprocesador(x: pd.DataFrame, escalar: bool) -> ColumnTransformer:
    numericas = x.select_dtypes(include=np.number).columns.tolist()
    categoricas = [c for c in x if c not in numericas]
    pasos = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        pasos.append(("escalar", StandardScaler()))
    return ColumnTransformer([
        ("numericas", Pipeline(pasos), numericas),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categoricas),
    ])


def especificacion(x: pd.DataFrame, tarea: str):
    if tarea == "binaria":
        return (
            Pipeline([
                ("pre", preprocesador(x, escalar=True)),
                ("modelo", LogisticRegression(max_iter=2000, random_state=SEED)),
            ]),
            {"modelo__C": [0.01, 0.1, 1, 10]},
            "roc_auc",
        )
    return (
        Pipeline([
            ("pre", preprocesador(x, escalar=False)),
            ("modelo", DecisionTreeClassifier(random_state=SEED)),
        ]),
        {
            "modelo__max_depth": [4, 8, 12, None],
            "modelo__min_samples_leaf": [10, 50],
        },
        "f1_macro",
    )


def metricas(tarea, y, pred, proba, clases):
    if tarea == "binaria":
        p = proba[:, list(clases).index(1)]
        return {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p),
            "average_precision": average_precision_score(y, p),
        }
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc_macro_ovr": roc_auc_score(
            y, proba, multi_class="ovr", average="macro", labels=clases
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    if df.shape != (50000, 28) or df.isna().any().any():
        raise ValueError("La forma o faltantes no coinciden con la base auditada")
    if set(df.target.unique()) != {"Graduado", "En Curso", "Desertor"}:
        raise ValueError("Etiquetas del target inesperadas")
    x = df.drop(columns="target")
    grupos = etapas(x.columns.tolist())
    # Una única división para las DOS tareas y TODAS las etapas.
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.20, stratify=df.target, random_state=SEED
    )
    xtr, xte = x.iloc[idx_train], x.iloc[idx_test]
    ytr_multi, yte_multi = df.target.iloc[idx_train], df.target.iloc[idx_test]
    huellas_train = set(pd.util.hash_pandas_object(xtr, index=False))
    no_visto_exacto = ~pd.util.hash_pandas_object(xte, index=False).isin(huellas_train).to_numpy()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    filas, cv_folds, matrices, por_clase = [], [], [], []
    predicciones = pd.DataFrame({
        "fila_original": idx_test,
        "target": yte_multi.to_numpy(),
        "sin_coincidencia_exacta_en_train": no_visto_exacto,
    })

    for tarea in ("binaria", "multiclase"):
        ytr = (ytr_multi == "Desertor").astype(int) if tarea == "binaria" else ytr_multi
        yte = (yte_multi == "Desertor").astype(int) if tarea == "binaria" else yte_multi
        for etapa, cols in grupos.items():
            modelo, grilla, criterio = especificacion(xtr[cols], tarea)
            print(f"{tarea}: {etapa} ({len(cols)} predictores); CV {criterio}", flush=True)
            busqueda = GridSearchCV(
                modelo, grilla, scoring=criterio, cv=cv,
                n_jobs=2, pre_dispatch=2, refit=True, error_score="raise",
            )
            busqueda.fit(xtr[cols], ytr)
            pred = busqueda.predict(xte[cols])
            proba = busqueda.predict_proba(xte[cols])
            clases = busqueda.classes_
            base = {"tarea": tarea, "etapa": etapa, "n_predictores": len(cols),
                    "criterio_cv": criterio, "cv_media": busqueda.best_score_,
                    "best_params": json.dumps(busqueda.best_params_, sort_keys=True)}
            filas.append({**base, "subconjunto": "test_completo",
                          "n_test": len(yte), **metricas(tarea, yte, pred, proba, clases)})
            filas.append({**base, "subconjunto": "sin_vector_exacto_en_train",
                          "n_test": int(no_visto_exacto.sum()),
                          **metricas(tarea, yte.iloc[no_visto_exacto],
                                     pred[no_visto_exacto], proba[no_visto_exacto], clases)})
            mejor_idx = busqueda.best_index_
            for fold in range(cv.n_splits):
                cv_folds.append({"tarea": tarea, "etapa": etapa, "fold": fold + 1,
                                 "criterio": criterio,
                                 "score_validacion": busqueda.cv_results_[f"split{fold}_test_score"][mejor_idx]})
            etiquetas = [0, 1] if tarea == "binaria" else ["Desertor", "En Curso", "Graduado"]
            cm = confusion_matrix(yte, pred, labels=etiquetas)
            for i, real in enumerate(etiquetas):
                for j, estimada in enumerate(etiquetas):
                    matrices.append({"tarea": tarea, "etapa": etapa, "real": real,
                                     "predicha": estimada, "n": int(cm[i, j])})
            if tarea == "multiclase":
                for clase in etiquetas:
                    sel = yte.to_numpy() == clase
                    por_clase.append({"etapa": etapa, "clase": clase, "n": int(sel.sum()),
                                      "precision": precision_score(yte, pred, labels=[clase], average="macro", zero_division=0),
                                      "recall": recall_score(yte, pred, labels=[clase], average="macro", zero_division=0),
                                      "f1": f1_score(yte, pred, labels=[clase], average="macro", zero_division=0)})
            predicciones[f"{tarea}_{etapa}"] = pred
            if tarea == "binaria":
                predicciones[f"p_desertor_{etapa}"] = proba[:, list(clases).index(1)]
            print(f"  CV={busqueda.best_score_:.4f}; test={filas[-2].get('roc_auc', filas[-2].get('f1_macro')):.4f}", flush=True)

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(xtr, ytr)
        dpred, dproba = dummy.predict(xte), dummy.predict_proba(xte)
        filas.append({"tarea": tarea, "etapa": "Dummy_mas_frecuente", "n_predictores": 0,
                      "criterio_cv": "—", "cv_media": np.nan, "best_params": "{}",
                      "subconjunto": "test_completo", "n_test": len(yte),
                      **metricas(tarea, yte, dpred, dproba, dummy.classes_)})

    pd.DataFrame(filas).to_csv(args.out / "metricas.csv", index=False)
    pd.DataFrame(cv_folds).to_csv(args.out / "validacion_folds.csv", index=False)
    pd.DataFrame(matrices).to_csv(args.out / "matrices_confusion.csv", index=False)
    pd.DataFrame(por_clase).to_csv(args.out / "metricas_por_clase.csv", index=False)
    predicciones.to_csv(args.out / "predicciones_test.csv", index=False)
    config = {"sha256_datos": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "semilla": SEED, "train": len(idx_train), "test": len(idx_test),
              "test_con_vector_exacto_en_train": int((~no_visto_exacto).sum()),
              "etapas": grupos, "advertencia": "test de esta base ya fue visto en el TP original; comparación exploratoria"}
    (args.out / "protocolo_ejecutado.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Guardado en", args.out)


if __name__ == "__main__":
    main()
