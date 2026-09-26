#!/usr/bin/env python3
"""EDA reproducible del TP1, después de reservar test y sólo sobre train.

Uso: python eda_train.py --input /ruta/dataset_TP1_2026.csv
Las salidas se escriben en `resultados/` junto al script por defecto.
No modifica ni carga archivos del TP original.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import scipy
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split


TARGET = "target"
CLASSES = ["Desertor", "En Curso", "Graduado"]
FIRST = {
    "Unidades_curriculares_1er_semestre_(inscrito)",
    "Cantidad_de_evaluaciones_en_el_1er_semestre",
    "Unidades_curriculares_1er_semestre_(aprobadas)",
    "Nota_promedio_en_el_1er_semestre",
}
SECOND = {
    "Unidades_curriculares_2do_semestre_(inscrito)",
    "Cantidad_de_evaluaciones_en_el_2do_semestre_",
    "Unidades_curriculares_2do_semestre_(aprobadas)",
    "Nota_promedio_en_el_2do_semestre",
}
ENTRY_SUGGESTED = {
    "Estudios_máximos_antes_de_la_inscripción",
    "modo_aplicacion",
    "Carrera",
    "Cualificación_promedio_de_estudios_previos",
    "Puntaje_en_examen_de_ingreso",
    "Edad_al_momento_de_la_inscripción",
    "estado_civil",
}
ENTRY_CANDIDATE = {
    "sexo",
    "desplazado",
    "Asistencia_diurna/nocturna",
    "Necesidades_educativas_especiales",
    "Estudiante_Internacional",
}
AMBIGUOUS = {
    "Tiene_deuda",
    "Pago_de_matrícula_al_día",
    "Poseedor_de_Beca",
    "Demanda_Cog_Padre",
    "Demanda_Cog_Madre",
    "Ingresos_Padre",
    "Ingresos_Madre",
}


def phase(column: str) -> str:
    if column in FIRST:
        return "cierre_1er_semestre_inferido"
    if column in SECOND:
        return "cierre_2do_semestre_inferido"
    if column in ENTRY_SUGGESTED:
        return "ingreso_sugerido_por_nombre_o_diccionario"
    if column in ENTRY_CANDIDATE:
        return "ingreso_candidato_fecha_por_confirmar"
    if column in AMBIGUOUS:
        return "fecha_de_captura_por_verificar"
    return "sin_clasificar"


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    z2 = z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    spread = z * np.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / (1 + z2 / total)
    return max(0, center - spread), min(1, center + spread)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_csv(df: pd.DataFrame, directory: Path, name: str) -> None:
    df.to_csv(directory / name, index=False, float_format="%.6g")


def one_row(variable: str, kind: str, phase_name: str, n: int, missing: int, n_unique: int, x: pd.Series) -> dict:
    return {
        "variable": variable,
        "tipo": kind,
        "momento_inferido": phase_name,
        "n_train": n,
        "faltantes": missing,
        "pct_faltantes": 100 * missing / n,
        "valores_distintos": n_unique,
        "pct_ceros_si_numerica": 100 * (x == 0).mean() if kind == "numerica" else np.nan,
        "min_si_numerica": x.min() if kind == "numerica" else np.nan,
        "max_si_numerica": x.max() if kind == "numerica" else np.nan,
    }


def categorical_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        # Include missing levels explicitly if this dataset ever changes.
        levels = df[col].astype("string").fillna("(faltante)")
        for level, index in levels.groupby(levels, dropna=False).groups.items():
            y = df.loc[index, TARGET]
            n = len(y)
            counts = y.value_counts()
            d = int(counts.get("Desertor", 0))
            lo, hi = wilson(d, n)
            rows.append({
                "variable": col,
                "momento_inferido": phase(col),
                "categoria": str(level),
                "n": n,
                "pct_del_train": 100 * n / len(df),
                "n_desertor": d,
                "pct_desertor": 100 * d / n,
                "ic95_desertor_inf_pct": 100 * lo,
                "ic95_desertor_sup_pct": 100 * hi,
                "pct_en_curso": 100 * counts.get("En Curso", 0) / n,
                "pct_graduado": 100 * counts.get("Graduado", 0) / n,
                "categoria_escasa_menos_100": n < 100,
            })
    return pd.DataFrame(rows).sort_values(["variable", "n"], ascending=[True, False])


def numeric_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        for label, sub in [("TOTAL_TRAIN", df), *[(k, df[df[TARGET] == k]) for k in CLASSES]]:
            x = sub[col].dropna()
            positive = x[x > 0]
            rows.append({
                "variable": col,
                "momento_inferido": phase(col),
                "grupo": label,
                "n": len(sub),
                "n_validos": len(x),
                "n_ceros": int((x == 0).sum()),
                "pct_ceros": 100 * (x == 0).mean(),
                "mediana_todos": x.median(),
                "q25_todos": x.quantile(.25),
                "q75_todos": x.quantile(.75),
                "media_todos": x.mean(),
                "n_positivos": len(positive),
                "mediana_positivos": positive.median(),
                "q25_positivos": positive.quantile(.25),
                "q75_positivos": positive.quantile(.75),
            })
    return pd.DataFrame(rows)


def associations(df: pd.DataFrame, numeric: list[str], categoric: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric:
        valid = df[[col, TARGET]].dropna()
        x = valid[col].to_numpy(dtype=float)
        overall = x.mean()
        ss_total = float(np.sum((x - overall) ** 2))
        ss_between = sum(len(g) * (g[col].mean() - overall) ** 2 for _, g in valid.groupby(TARGET))
        # η² measures descriptive class separation within train; it is not out-of-sample importance.
        rows.append({"variable": col, "momento_inferido": phase(col),
                     "metodo": "eta2_tres_clases", "magnitud_train": ss_between / ss_total if ss_total else 0.,
                     "n_train": len(valid), "nota": "Asociacion marginal descriptiva; no efecto causal"})
    for col in categoric:
        valid = df[[col, TARGET]].copy()
        tab = pd.crosstab(valid[col].astype("string").fillna("(faltante)"), valid[TARGET])
        chi2, _, _, _ = chi2_contingency(tab, correction=False)
        denom = len(valid) * min(tab.shape[0] - 1, tab.shape[1] - 1)
        rows.append({"variable": col, "momento_inferido": phase(col),
                     "metodo": "cramers_v_tres_clases", "magnitud_train": np.sqrt(chi2 / denom) if denom else 0.,
                     "n_train": len(valid), "nota": "Asociacion marginal descriptiva; V no equiparable a eta2"})
    return pd.DataFrame(rows).sort_values(["metodo", "magnitud_train"], ascending=[True, False])


def redundancy(df: pd.DataFrame, numeric: list[str], categoric: list[str]) -> pd.DataFrame:
    rows = []
    correlations = df[numeric].corr(method="spearman")
    for i, a in enumerate(numeric):
        for b in numeric[i+1:]:
            rho = correlations.loc[a, b]
            if abs(rho) >= .65:
                rows.append({"variable_a": a, "variable_b": b,
                             "relacion": "spearman", "magnitud": rho,
                             "detalle": "Asociacion de rangos; no implica duplicacion"})
    for i, a in enumerate(categoric):
        for b in categoric[i+1:]:
            a_unique_per_b = df.groupby(b, dropna=False)[a].nunique(dropna=False).max()
            b_unique_per_a = df.groupby(a, dropna=False)[b].nunique(dropna=False).max()
            if a_unique_per_b == 1 and b_unique_per_a == 1:
                rows.append({"variable_a": a, "variable_b": b,
                             "relacion": "biyeccion_categorica_en_train", "magnitud": 1,
                             "detalle": "Cada categoria de una determina exactamente la otra en train"})
    return pd.DataFrame(rows, columns=["variable_a", "variable_b", "relacion", "magnitud", "detalle"])


def semester_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f_enrolled = df["Unidades_curriculares_1er_semestre_(inscrito)"]
    f_approved = df["Unidades_curriculares_1er_semestre_(aprobadas)"]
    s_enrolled = df["Unidades_curriculares_2do_semestre_(inscrito)"]
    s_approved = df["Unidades_curriculares_2do_semestre_(aprobadas)"]
    predictors = [c for c in df if c != TARGET]
    conflict = df.groupby(predictors, dropna=False)[TARGET].agg(["nunique", "size"])
    conflict = conflict[conflict["nunique"] > 1]
    quality = pd.DataFrame([
        {"chequeo": "primer_semestre_aprobadas_mayor_inscritas", "n": int((f_approved > f_enrolled).sum())},
        {"chequeo": "segundo_semestre_aprobadas_mayor_inscritas", "n": int((s_approved > s_enrolled).sum())},
        {"chequeo": "primer_semestre_sin_inscritas_y_con_aprobadas", "n": int(((f_enrolled == 0) & (f_approved > 0)).sum())},
        {"chequeo": "segundo_semestre_sin_inscritas_y_con_aprobadas", "n": int(((s_enrolled == 0) & (s_approved > 0)).sum())},
        {"chequeo": "filas_duplicadas_exactas_con_target", "n": int(df.duplicated().sum())},
        {"chequeo": "filas_duplicadas_en_predictores", "n": int(df.drop(columns=TARGET).duplicated().sum())},
        {"chequeo": "patrones_de_predictores_con_targets_distintos", "n": len(conflict)},
        {"chequeo": "filas_en_patrones_con_targets_distintos", "n": int(conflict["size"].sum())},
    ])
    bins1 = np.select([f_enrolled.eq(0), f_approved.eq(0)], ["0_inscritas", "0_aprobadas_con_inscripcion"], default="al_menos_1_aprobada")
    bins2 = np.select([s_enrolled.eq(0), s_approved.eq(0)], ["0_inscritas", "0_aprobadas_con_inscripcion"], default="al_menos_1_aprobada")
    tmp = df[[TARGET]].copy()
    tmp["primer_semestre"] = bins1
    tmp["segundo_semestre"] = bins2
    rows = []
    for (first, second), g in tmp.groupby(["primer_semestre", "segundo_semestre"]):
        n = len(g)
        d = int((g[TARGET] == "Desertor").sum())
        lo, hi = wilson(d, n)
        rows.append({"primer_semestre": first, "segundo_semestre": second,
                     "n": n, "pct_desertor": 100 * d / n,
                     "ic95_desertor_inf_pct": 100 * lo, "ic95_desertor_sup_pct": 100 * hi,
                     "pct_en_curso": 100 * (g[TARGET] == "En Curso").mean(),
                     "pct_graduado": 100 * (g[TARGET] == "Graduado").mean()})
    return quality, pd.DataFrame(rows).sort_values(["primer_semestre", "segundo_semestre"])


def stratified_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Descripciones por carga aprobada y otra variable; cortes exploratorios, no optimizados."""
    x = df["Unidades_curriculares_1er_semestre_(aprobadas)"]
    strata = pd.cut(x, bins=[-np.inf, 0, 3, np.inf],
                    labels=["0_aprobadas", "1_a_3_aprobadas", "4_o_mas_aprobadas"])
    rows = []
    for col in ["Pago_de_matrícula_al_día", "Tiene_deuda", "Poseedor_de_Beca", "Carrera"]:
        for (stratum, category), pos in df.groupby([strata, df[col]], observed=True).groups.items():
            y = df.loc[pos, TARGET]
            n = len(y)
            d = int((y == "Desertor").sum())
            lo, hi = wilson(d, n)
            rows.append({"aprobadas_1er_semestre_grupo": str(stratum), "variable": col,
                         "categoria": str(category), "n": n, "n_desertor": d,
                         "pct_desertor": 100 * d / n, "ic95_inf_pct": 100 * lo,
                         "ic95_sup_pct": 100 * hi, "escasa_menos_100": n < 100})
    return pd.DataFrame(rows).sort_values(["variable", "aprobadas_1er_semestre_grupo", "n"],
                                          ascending=[True, True, False])


def plot(df: pd.DataFrame, categorical: pd.DataFrame, trajectory: pd.DataFrame, out: Path) -> None:
    # Paleta editorial acotada: rojo destaca Desertor; los otros estados van en gris.
    red, dark, mid, light = "#A92F36", "#242424", "#777777", "#C5C5C5"
    colors = [red, mid, light]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.spines.bottom": False,
                         "axes.labelcolor": dark, "text.color": dark,
                         "xtick.color": mid, "ytick.color": dark,
                         "figure.facecolor": "white", "axes.facecolor": "white"})

    def layout(fig, ax, title, subtitle, note, left=.27):
        fig.subplots_adjust(left=left, right=.91, top=.71, bottom=.19)
        fig.text(left, .95, title, fontsize=16, fontweight="bold", ha="left")
        fig.text(left, .875, subtitle, fontsize=10, color=mid, ha="left")
        fig.add_artist(plt.Line2D([left, left + .075], [.835, .835],
                                  transform=fig.transFigure, lw=3, color=red))
        fig.text(left, .06, note, fontsize=9, color=mid, ha="left")
        ax.tick_params(length=0, pad=8)
        ax.set_axisbelow(True)

    counts = df[TARGET].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(CLASSES))
    ax.barh(y, counts / len(df) * 100, height=.57, color=colors)
    ax.set_yticks(y, CLASSES)
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xticks([])
    for i, v in enumerate(counts):
        ax.text(100 * v / len(df) + 1, i,
                f"{v / len(df):.1%}  ·  n={v:,}", va="center", fontsize=11)
    layout(fig, ax, "Tres desenlaces, tamaños distintos",
           "Distribución de clases en 40.000 registros de entrenamiento",
           "Base: train. Test reservado para evaluación final.", left=.27)
    fig.savefig(out / "01_clases_train.png", dpi=180); plt.close(fig)

    # Un solo indicador por semestre: el cero combina falta de actividad o aprobación.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, col, label in zip(axes,
                              ["Unidades_curriculares_1er_semestre_(aprobadas)",
                               "Unidades_curriculares_2do_semestre_(aprobadas)"],
                              ["1.er semestre", "2.º semestre"]):
        values = [100 * df.loc[df[TARGET] == group, col].eq(0).mean() for group in CLASSES]
        y = np.arange(len(CLASSES))
        ax.barh(y, values, height=.48, color=colors)
        ax.set_yticks(y, CLASSES)
        ax.invert_yaxis()
        ax.set_xlim(0, 83)
        ax.set_xticks([])
        ax.set_title(label, loc="left", fontsize=12, pad=13, fontweight="bold")
        for i, v in enumerate(values):
            ax.text(v + 2, i, f"{v:.1f}%", va="center")
        ax.tick_params(length=0, pad=5)
    fig.subplots_adjust(left=.12, right=.96, wspace=.44, top=.69, bottom=.20)
    fig.text(.12, .95, "Los ceros separan fuertemente a los grupos", fontsize=16, fontweight="bold")
    fig.text(.12, .875, "Estudiantes con 0 materias aprobadas, dentro de cada clase", fontsize=10, color=mid)
    fig.add_artist(plt.Line2D([.12, .18], [.835, .835], transform=fig.transFigure, lw=3, color=red))
    fig.text(.12, .06, "Un 0 no distingue inactividad, falta de aprobación ni fecha de abandono.", fontsize=9, color=mid)
    fig.savefig(out / "02_ceros_academicos.png", dpi=180); plt.close(fig)

    # Intervalos intercuartílicos: una mediana sola ocultaría el solapamiento.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1))
    for ax, col, label, limits in zip(
        axes,
        ["Edad_al_momento_de_la_inscripción", "Puntaje_en_examen_de_ingreso"],
        ["Edad al inscribirse · años", "Puntaje de ingreso"],
        [(14, 43), (40, 87)],
    ):
        for i, (group, color) in enumerate(zip(CLASSES, colors)):
            x = df.loc[df[TARGET] == group, col].dropna()
            q1, med, q3 = x.quantile([.25, .5, .75])
            ax.plot([q1, q3], [i, i], color=color, linewidth=7, solid_capstyle="round")
            ax.scatter([med], [i], s=42, color=dark, zorder=3)
            ax.text(q3 + 1, i, f"{med:.0f}", va="center", fontsize=10)
        ax.set_xlim(*limits)
        ax.set_yticks(range(3), CLASSES)
        ax.invert_yaxis()
        ax.set_title(label, loc="left", fontsize=12, pad=14, fontweight="bold")
        ax.tick_params(length=0, pad=6)
    fig.subplots_adjust(left=.12, right=.96, wspace=.38, top=.69, bottom=.20)
    fig.text(.12, .95, "Antes del primer semestre también hay diferencias", fontsize=16, fontweight="bold")
    fig.text(.12, .875, "Rango del 25.º al 75.º percentil; punto negro = mediana", fontsize=10, color=mid)
    fig.add_artist(plt.Line2D([.12, .18], [.835, .835], transform=fig.transFigure, lw=3, color=red))
    fig.text(.12, .06, "Cifras descriptivas; para la distribución completa ver 03_numericas_por_clase_train.csv.", fontsize=9, color=mid)
    fig.savefig(out / "03_puntajes_ingreso.png", dpi=180); plt.close(fig)

    # Seis categorías de tres variables con fecha de captura desconocida.
    sub = categorical[categorical.variable.isin(
        ["Tiene_deuda", "Pago_de_matrícula_al_día", "Poseedor_de_Beca"])]
    names = [("Pago_de_matrícula_al_día", "No", "Matrícula no al día"),
             ("Pago_de_matrícula_al_día", "Sí", "Matrícula al día"),
             ("Tiene_deuda", "Sí", "Con deuda"),
             ("Tiene_deuda", "No", "Sin deuda"),
             ("Poseedor_de_Beca", "Sí", "Con beca"),
             ("Poseedor_de_Beca", "No", "Sin beca")]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    baseline = 100 * (df[TARGET] == "Desertor").mean()
    ax.axvline(baseline, color=light, linestyle="--", linewidth=1.5)
    ax.text(baseline + .7, -.52, f"Total {baseline:.1f}%", fontsize=9, color=mid)
    for i, (column, category, label) in enumerate(names):
        row = sub[(sub.variable == column) & (sub.categoria == category)].iloc[0]
        color = red if row.pct_desertor > baseline else mid
        ax.barh(i, row.pct_desertor, height=.48, color=color)
        ax.text(row.pct_desertor + 1, i, f"{row.pct_desertor:.1f}%  ·  n={row.n:,}", va="center", fontsize=10)
    ax.set_yticks(np.arange(len(names)), [label for _, _, label in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 117)
    ax.set_xticks([])
    layout(fig, ax, "Deuda, matrícula y beca: tasas diferentes",
           "Porcentaje Desertor dentro de cada categoría · línea gris: promedio de train",
           "Fecha de registro desconocida. Las diferencias no prueban un efecto causal.", left=.28)
    fig.savefig(out / "04_categorias_financieras.png", dpi=180); plt.close(fig)

    # Sólo los cuatro patrones con inscripción en ambos semestres y n >= 100.
    pairs = [("0_aprobadas_con_inscripcion", "0_aprobadas_con_inscripcion", "0 → 0"),
             ("al_menos_1_aprobada", "0_aprobadas_con_inscripcion", "1+ → 0"),
             ("0_aprobadas_con_inscripcion", "al_menos_1_aprobada", "0 → 1+"),
             ("al_menos_1_aprobada", "al_menos_1_aprobada", "1+ → 1+")]
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    for i, (first, second, label) in enumerate(pairs):
        row = trajectory[(trajectory.primer_semestre == first) &
                         (trajectory.segundo_semestre == second)].iloc[0]
        ax.barh(i, row.pct_desertor, height=.52, color=red if row.pct_desertor > 80 else mid)
        ax.text(row.pct_desertor + 1.2, i, f"{row.pct_desertor:.1f}%  ·  n={row.n:,}", va="center")
    ax.set_yticks(np.arange(4), [label for _, _, label in pairs])
    ax.invert_yaxis()
    ax.set_xlim(0, 122)
    ax.set_xticks([])
    layout(fig, ax, "La aprobación en ambos semestres distingue grupos",
           "Desertor por patrón de materias aprobadas · 1.er → 2.º semestre",
           "Sólo inscriptos en ambos semestres; n≥100. Se desconoce cuándo ocurrió la deserción.", left=.20)
    fig.savefig(out / "05_trayectorias_academicas.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "resultados")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input)
    if TARGET not in raw or set(raw[TARGET]) != set(CLASSES):
        raise ValueError("Se esperan target y las clases Desertor, En Curso y Graduado")
    tr_idx, test_idx = train_test_split(np.arange(len(raw)), test_size=.20,
                                       random_state=42, stratify=raw[TARGET])
    df = raw.iloc[tr_idx].copy()  # Only train is inspected below. Test remains locked.
    numeric = df.drop(columns=TARGET).select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df if c != TARGET and c not in numeric]
    n = len(df)
    manifest = {"source_file_name": args.input.name,
                "source_sha256": sha256(args.input), "n_total_metadata": len(raw),
                "n_train": n, "n_test_reservado": len(test_idx),
                "split": {"test_size": .2, "stratify": "target", "random_state": 42,
                          "sklearn_train_test_split": True},
                "train_index_sha256": hashlib.sha256(np.sort(tr_idx).astype("<i8").tobytes()).hexdigest(),
                "no_test_target_analysis": True,
                "versiones": {"pandas": pd.__version__, "numpy": np.__version__,
                              "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                              "matplotlib": matplotlib.__version__},
                "numericas": numeric, "categoricas": categorical,
                "nota": "La exploración del target y predictores sólo usa train; test se reserva para evaluación final."}
    (args.output / "manifiesto_split.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    base = df[TARGET].value_counts().reindex(CLASSES).rename_axis("clase").reset_index(name="n")
    base["pct_train"] = 100 * base.n / n
    save_csv(base, args.output, "01_clases_train.csv")
    quality = [one_row(c, "numerica" if c in numeric else "categorica", phase(c),
                       n, int(df[c].isna().sum()), int(df[c].nunique(dropna=False)), df[c])
               for c in df if c != TARGET]
    save_csv(pd.DataFrame(quality), args.output, "02_inventario_y_calidad_train.csv")
    nstats = numeric_table(df, numeric)
    save_csv(nstats, args.output, "03_numericas_por_clase_train.csv")
    cstats = categorical_table(df, categorical)
    save_csv(cstats, args.output, "04_categoricas_por_clase_train.csv")
    save_csv(associations(df, numeric, categorical), args.output, "05_asociaciones_marginales_train.csv")
    save_csv(redundancy(df, numeric, categorical), args.output, "06_redundancia_train.csv")
    checks, trajectory = semester_table(df)
    save_csv(checks, args.output, "07_chequeos_train.csv")
    save_csv(trajectory, args.output, "08_trayectorias_academicas_train.csv")
    save_csv(stratified_rates(df), args.output, "09_tasas_estratificadas_train.csv")
    plot(df, cstats, trajectory, args.output)
    print(f"EDA completado en {args.output.resolve()} | train={n} | test reservado={len(test_idx)}")


if __name__ == "__main__":
    main()
