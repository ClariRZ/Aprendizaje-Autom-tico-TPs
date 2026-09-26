from pathlib import Path
import sys
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from _comun import OUT

# ---------------------------------------------------------------------------
# Este bloque NO entrena modelos.
# Integra únicamente resultados ya obtenidos.
# ---------------------------------------------------------------------------

def exigir(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path.name}. Ejecutá primero el bloque correspondiente."
        )
    return pd.read_csv(path)

# ---------------------------------------------------------------------------
# 1. Comparación binaria
# ---------------------------------------------------------------------------
bin_base = exigir(OUT / "03_binario.csv")
voting = exigir(OUT / "05_voting.csv")

cols_bin = [
    "modelo", "accuracy", "precision", "recall", "f1", "roc_auc"
]

binaria = pd.concat(
    [
        bin_base[cols_bin].assign(bloque_origen="03_modelos_binarios"),
        voting[cols_bin].assign(bloque_origen="05_voting"),
    ],
    ignore_index=True,
)

# Agregamos CV/timing donde existen, sin fingir que Voting tuvo una búsqueda
# comparable a los modelos base.
extra_bin = bin_base[
    [c for c in ["modelo", "cv_roc_auc", "tiempo_min"] if c in bin_base.columns]
].copy()

if "cv_roc_auc" not in extra_bin.columns:
    extra_bin["cv_roc_auc"] = pd.NA
if "tiempo_min" not in extra_bin.columns:
    extra_bin["tiempo_min"] = pd.NA

binaria = binaria.merge(extra_bin, on="modelo", how="left", suffixes=("", "_base"))

# Tiempo de Voting
if "tiempo_min" in voting.columns:
    mask = binaria["modelo"].eq("Voting soft")
    binaria.loc[mask, "tiempo_min"] = float(voting.iloc[0]["tiempo_min"])

binaria["gap_cv_test_auc"] = (
    binaria["cv_roc_auc"] - binaria["roc_auc"]
)

binaria = binaria.sort_values("roc_auc", ascending=False).reset_index(drop=True)
binaria["orden_auc_test"] = binaria.index + 1

binaria.to_csv(OUT / "09_comparacion_binaria.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Comparación multiclase
# ---------------------------------------------------------------------------
multi_base = exigir(OUT / "06_multiclase.csv")
ensembles = exigir(OUT / "07_ensembles.csv")

cols_multi = [
    "modelo",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc_macro_ovr",
    "cv_f1_macro",
    "tiempo_min",
]

multiclase = pd.concat(
    [
        multi_base[cols_multi].assign(bloque_origen="06_multiclase"),
        ensembles[cols_multi].assign(bloque_origen="07_ensembles"),
    ],
    ignore_index=True,
)

multiclase["gap_cv_test_f1"] = (
    multiclase["cv_f1_macro"] - multiclase["f1_macro"]
)

multiclase = multiclase.sort_values(
    "f1_macro", ascending=False
).reset_index(drop=True)
multiclase["orden_f1_macro_test"] = multiclase.index + 1

multiclase.to_csv(OUT / "09_comparacion_multiclase.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Resumen textual
# ---------------------------------------------------------------------------
def fmt(x, n=4):
    if pd.isna(x):
        return "—"
    return f"{float(x):.{n}f}"

lines = [
    "# Comparación final de modelos",
    "",
    "Este resumen mantiene separadas las tareas binaria y multiclase.",
    "No se comparan directamente scores entre tareas diferentes.",
    "",
    "## Clasificación binaria",
    "",
    "| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC | Gap CV-test |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]

for _, r in binaria.iterrows():
    lines.append(
        f"| {r['modelo']} | {fmt(r['accuracy'])} | {fmt(r['precision'])} | "
        f"{fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['roc_auc'])} | "
        f"{fmt(r['cv_roc_auc'])} | {fmt(r['gap_cv_test_auc'])} |"
    )

lines += [
    "",
    "### Lectura binaria",
    "",
    "- La comparación de modelos individuales se apoya en las mismas 10.000 observaciones de test.",
    "- Voting usa el mismo test pero no tiene una búsqueda de hiperparámetros equivalente, por lo que su columna CV queda vacía.",
    "- ROC-AUC y las métricas con umbral 0,5 responden a preguntas distintas; no se fuerza un único modelo como superior en todas las dimensiones.",
    "",
    "## Clasificación multiclase",
    "",
    "| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | ROC-AUC macro OvR | CV F1 macro | Gap CV-test |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]

for _, r in multiclase.iterrows():
    lines.append(
        f"| {r['modelo']} | {fmt(r['accuracy'])} | {fmt(r['precision_macro'])} | "
        f"{fmt(r['recall_macro'])} | {fmt(r['f1_macro'])} | "
        f"{fmt(r['roc_auc_macro_ovr'])} | {fmt(r['cv_f1_macro'])} | "
        f"{fmt(r['gap_cv_test_f1'])} |"
    )

lines += [
    "",
    "### Lectura multiclase",
    "",
    "- Todos los modelos se evaluaron sobre las mismas tres clases y el mismo test.",
    "- F1 macro es la métrica central de selección porque otorga igual peso a Graduado, En Curso y Desertor.",
    "- Los ensambles se comparan con el Árbol individual para estudiar el aporte de Bagging y Boosting.",
    "",
    "## Regla metodológica",
    "",
    "Este bloque integra resultados ya cerrados. No reentrena, no retunea y no usa la comparación final para reabrir hiperparámetros.",
]

resumen = OUT / "09_resumen_comparacion.md"
resumen.write_text("\n".join(lines), encoding="utf-8")

print("\n" + "=" * 80)
print("COMPARACIÓN BINARIA")
print("=" * 80)
print(
    binaria[
        [
            "modelo", "accuracy", "precision", "recall", "f1",
            "roc_auc", "cv_roc_auc", "gap_cv_test_auc"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("COMPARACIÓN MULTICLASE")
print("=" * 80)
print(
    multiclase[
        [
            "modelo", "accuracy", "precision_macro", "recall_macro",
            "f1_macro", "roc_auc_macro_ovr", "cv_f1_macro",
            "gap_cv_test_f1"
        ]
    ].to_string(index=False)
)

print("\nGuardado:")
print(OUT / "09_comparacion_binaria.csv")
print(OUT / "09_comparacion_multiclase.csv")
print(resumen)
