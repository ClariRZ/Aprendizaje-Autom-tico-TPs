#!/usr/bin/env python3
"""Atlas univariado y bivariado completo, calculado exclusivamente en train.

Ejecutar desde la raíz del repositorio:
python "AA privada/02_alternativa_temporal/eda/atlas_variables.py" \
  --input 00_base_de_datos/dataset_TP1_2026.csv

Genera 27 SVG y un índice legible. No entrena un modelo ni mira el target de test.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from eda_train import CLASSES, TARGET, FIRST, SECOND, phase, sha256


RED = "#A92F36"
DARK = "#272727"
MID = "#777777"
LIGHT = "#C5C5C5"
PALETTE = [RED, MID, LIGHT]
PREFIX = {
    "Cualificación_promedio_de_estudios_previos": "Nota de estudios previos",
    "Puntaje_en_examen_de_ingreso": "Puntaje de ingreso",
    "Edad_al_momento_de_la_inscripción": "Edad al inscribirse",
    "Unidades_curriculares_1er_semestre_(inscrito)": "Materias inscriptas · 1.er semestre",
    "Cantidad_de_evaluaciones_en_el_1er_semestre": "Evaluaciones · 1.er semestre",
    "Unidades_curriculares_1er_semestre_(aprobadas)": "Materias aprobadas · 1.er semestre",
    "Nota_promedio_en_el_1er_semestre": "Nota promedio · 1.er semestre",
    "Unidades_curriculares_2do_semestre_(inscrito)": "Materias inscriptas · 2.º semestre",
    "Cantidad_de_evaluaciones_en_el_2do_semestre_": "Evaluaciones · 2.º semestre",
    "Unidades_curriculares_2do_semestre_(aprobadas)": "Materias aprobadas · 2.º semestre",
    "Nota_promedio_en_el_2do_semestre": "Nota promedio · 2.º semestre",
    "Estudios_máximos_antes_de_la_inscripción": "Estudios anteriores",
    "estado_civil": "Estado civil al inscribirse",
    "modo_aplicacion": "Modo de ingreso",
    "sexo": "Sexo registrado",
    "desplazado": "Desplazamiento de localidad",
    "Carrera": "Carrera",
    "Asistencia_diurna/nocturna": "Turno",
    "Necesidades_educativas_especiales": "Necesidades educativas declaradas",
    "Estudiante_Internacional": "Estudiante internacional",
    "Tiene_deuda": "Deuda con la institución",
    "Pago_de_matrícula_al_día": "Matrícula al día",
    "Poseedor_de_Beca": "Beca",
    "Demanda_Cog_Padre": "Demanda cognitiva de ocupación · padre",
    "Demanda_Cog_Madre": "Demanda cognitiva de ocupación · madre",
    "Ingresos_Padre": "Nivel de ingreso estimado · padre",
    "Ingresos_Madre": "Nivel de ingreso estimado · madre",
}
ORDERED = {
    "Demanda_Cog_Padre": ["Muy baja demanda cognitiva", "Demanda cognitiva baja",
                            "Demanda cognitiva media", "Alta demanda cognitiva",
                            "Muy alta demanda cognitiva"],
    "Demanda_Cog_Madre": ["Muy baja demanda cognitiva", "Demanda cognitiva baja",
                            "Demanda cognitiva media", "Alta demanda cognitiva",
                            "Muy alta demanda cognitiva"],
    "Ingresos_Padre": ["Bajos ingresos", "Ingresos medios-bajos", "Ingresos medios",
                       "Ingresos medios-altos", "Altos ingresos"],
    "Ingresos_Madre": ["Bajos ingresos", "Ingresos medios-bajos", "Ingresos medios",
                       "Ingresos medios-altos", "Altos ingresos"],
}


def number(value: float) -> str:
    return f"{value:.1f}".replace(".0", "")


def fmt_n(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def theme() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": DARK, "axes.labelcolor": DARK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.bottom": False, "axes.spines.left": False,
        "xtick.color": MID, "ytick.color": DARK,
    })


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="svg", facecolor="white")
    plt.close(fig)


def numeric_figure(df: pd.DataFrame, col: str, path: Path) -> dict:
    x = df[col].dropna()
    academic = col in FIRST or col in SECOND
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 6.0),
                                      gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.08, right=.94, bottom=.34, top=.72, wspace=.30)
    fig.text(.08, .94, PREFIX[col], fontsize=18, fontweight="bold")
    fig.text(.08, .86, "Distribución de train y comparación descriptiva entre las tres clases",
             color=MID, fontsize=10)
    fig.add_artist(plt.Line2D([.08, .14], [.82, .82], transform=fig.transFigure,
                              lw=3, color=RED))
    # Histogram bins are integer-centered because the original columns contain integer values.
    vmin, vmax = int(x.min()), int(x.max())
    bins = np.arange(vmin - .5, vmax + 1.5)
    if len(bins) > 52:
        bins = 30
    left.hist(x, bins=bins, color=LIGHT, edgecolor="white", linewidth=.45)
    left.set(title="¿Qué valores hay?", xlabel="Valor", ylabel="Estudiantes")
    left.tick_params(length=0)
    if academic:
        left.axvline(0, color=RED, lw=2)
        left.text(.98, .95, f"Cero: {x.eq(0).mean():.1%}", color=RED,
                  transform=left.transAxes, ha="right", va="top", fontsize=10)

    desc = []
    for i, (name, color) in enumerate(zip(CLASSES, PALETTE)):
        group = df.loc[df[TARGET] == name, col].dropna()
        shown = group[group > 0] if academic else group
        if shown.empty:
            desc.append((name, len(group), float("nan"), float("nan")))
            continue
        q1, median, q3 = shown.quantile([.25, .5, .75])
        right.plot([q1, q3], [i, i], color=color, lw=7, solid_capstyle="round")
        right.scatter([median], [i], color=DARK, s=35, zorder=2)
        desc.append((name, len(group), float(median), 100 * group.eq(0).mean()))
    right.set_yticks(range(3), CLASSES)
    right.invert_yaxis()
    right.set_xlim(vmin - 1, vmax + max(2, (vmax - vmin) * .12))
    right.set(title="¿Cómo se distribuye por clase?", xlabel="Valor")
    right.tick_params(length=0, pad=7)
    fig.text(.08, .055,
             "Línea: 25.º–75.º percentil; punto: mediana. " +
             ("En el panel por clase se excluyen ceros: ver sus porcentajes abajo."
              if academic else "Grupos con tamaños distintos: no comparar alturas como tasas."),
             fontsize=9, color=MID)
    if academic:
        zero_note = "  ·  ".join(f"{name}: {zero:.1f}%" for name, _, _, zero in desc)
        med_note = "  ·  ".join(f"{name}: {number(median)}" for name, _, median, _ in desc)
        fig.text(.08, .20, "Ceros dentro de la clase: " + zero_note, fontsize=9, color=DARK)
        fig.text(.08, .125, "Mediana entre positivos: " + med_note, fontsize=9, color=DARK)
    else:
        note = "  ·  ".join(f"{name}: mediana={number(median)}" for name, _, median, _ in desc)
        fig.text(.08, .16, note, fontsize=9, color=DARK)
    save(fig, path)
    return {"desc": desc, "n": len(x), "missing": int(df[col].isna().sum()),
            "min": vmin, "max": vmax, "zeros": int(x.eq(0).sum()), "academic": academic}


def categorical_figure(df: pd.DataFrame, col: str, path: Path) -> dict:
    series = df[col].astype("string").fillna("(faltante)")
    ct = pd.crosstab(series, df[TARGET]).reindex(columns=CLASSES, fill_value=0)
    if col in ORDERED:
        levels = [str(v) for v in ORDERED[col] if v in ct.index]
        levels += [str(v) for v in ct.index if v not in levels]
    else:
        levels = list(ct.sum(axis=1).sort_values(ascending=False).index.astype(str))
    ct = ct.loc[levels]
    n = ct.sum(axis=1).astype(int)
    d_rate = 100 * ct["Desertor"] / n
    height = max(5.25, 3.15 + .43 * len(levels))
    fig, (left, right) = plt.subplots(1, 2, figsize=(14.4, height), sharey=True,
                                      gridspec_kw={"width_ratios": [1, 1.1]})
    top = 1 - 1.65 / height
    bottom = .70 / height
    fig.subplots_adjust(left=.33, right=.92, bottom=bottom, top=top, wspace=.12)
    fig.text(.33, 1 - .30 / height, PREFIX[col], fontsize=17, fontweight="bold")
    fig.text(.33, 1 - .73 / height,
             "¿Cuántos hay en cada categoría? ¿Qué porcentaje es Desertor dentro de ella?",
             color=MID, fontsize=10)
    y = np.arange(len(levels))
    left.barh(y, n.to_numpy(), height=.65, color=LIGHT)
    left.set_xlim(0, max(n) * 1.24)
    left.set_yticks(y, levels)
    left.invert_yaxis()
    left.set_title("Tamaño de la categoría", loc="left", fontsize=11, pad=13)
    left.set_xticks([])
    for i, n_level in enumerate(n):
        left.text(n_level + max(n) * .012, i, f"n={fmt_n(n_level)}",
                  va="center", fontsize=9)
    baseline = 100 * (df[TARGET] == "Desertor").mean()
    right.axvline(baseline, color=LIGHT, ls="--", lw=1.3)
    right.barh(y, d_rate.to_numpy(), height=.65,
               color=[RED if value > baseline and count >= 100 else MID
                      for value, count in zip(d_rate, n)])
    right.set_xlim(0, 118)
    right.set_title("Desertor dentro de la categoría", loc="left", fontsize=11, pad=13)
    right.set_xticks([])
    right.tick_params(axis="y", left=False, labelleft=False)
    for i, value in enumerate(d_rate):
        right.text(value + 1.5, i, f"{value:.1f}%", va="center", fontsize=9)
    left.tick_params(length=0, pad=5)
    fig.text(.33, .32 / height,
             f"Rojo: por encima del total ({baseline:.1f}%) y n≥100. "
             "Celdas pequeñas quedan en gris.", color=MID, fontsize=9)
    save(fig, path)
    rows = [{"level": lev, "n": int(n.loc[lev]), "desertor": int(ct.loc[lev, "Desertor"]),
             "rate": float(d_rate.loc[lev])} for lev in levels]
    return {"rows": rows, "missing": int(df[col].isna().sum()), "n_levels": len(rows)}


def explain_numeric(col: str, stats: dict) -> str:
    desc = stats["desc"]
    med = ", ".join(f"{name} {number(median)}" for name, _, median, _ in desc)
    if stats["academic"]:
        zeros = ", ".join(f"{name} {zero:.1f}%" for name, _, _, zero in desc)
        return (f"Valores: {number(stats['min'])}–{number(stats['max'])}; "
                f"cero en {fmt_n(stats['zeros'])}/{fmt_n(stats['n'])} observaciones. "
                f"**Ceros dentro de cada clase:** {zeros}. "
                f"**Mediana entre valores positivos:** {med}. "
                "Comparar también el número de observaciones positivas en la tabla, "
                "y verificar si cero significa falta de aprobación, evaluación o cursada.")
    return (f"Valores: {number(stats['min'])}–{number(stats['max'])}; "
            f"**medianas por clase:** {med}. La mediana sola no mide "
            "capacidad predictiva; observar solapamiento de distribuciones y tamaños.")


def explain_category(stats: dict) -> str:
    vals = stats["rows"]
    common = max(vals, key=lambda r: r["n"])
    eligible = [r for r in vals if r["n"] >= 100]
    if not eligible:
        return "Todas las categorías tienen menos de 100 casos; consultar tasas con gran incertidumbre."
    hi = max(eligible, key=lambda r: r["rate"])
    lo = min(eligible, key=lambda r: r["rate"])
    thin = sum(r["n"] < 100 for r in vals)
    warning = (f" **{thin} categoría(s) con menos de 100 casos:** no ordenarlas como si la tasa fuese exacta."
               if thin else "")
    return (f"**{stats['n_levels']} categorías.** La más frecuente es «{common['level']}» "
            f"(n={fmt_n(common['n'])}). Entre categorías con n≥100, la tasa Desertor va "
            f"de {lo['rate']:.1f}% en «{lo['level']}» (n={fmt_n(lo['n'])}) a "
            f"{hi['rate']:.1f}% en «{hi['level']}» (n={fmt_n(hi['n'])})."
            + warning + " Son asociaciones descriptivas, sin ajuste por otras variables.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "atlas")
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    if TARGET not in raw or set(raw[TARGET]) != set(CLASSES):
        raise ValueError("La base debe incluir target y las tres clases originales")
    tr, test = train_test_split(np.arange(len(raw)), test_size=.2, stratify=raw[TARGET],
                                random_state=42)
    df = raw.iloc[tr].copy()
    expected = set(df.columns) - {TARGET}
    if expected != set(PREFIX) or len(expected) != 27:
        raise ValueError(f"Cambió el inventario de columnas: faltan {expected - set(PREFIX)}; "
                         f"sobran {set(PREFIX) - expected}")
    numeric = df.drop(columns=TARGET).select_dtypes(include="number").columns.tolist()
    categoric = [c for c in df if c not in numeric and c != TARGET]
    if (len(numeric), len(categoric)) != (11, 16):
        raise ValueError("Cambió el tipo leído de las 27 variables; revisar diccionario")
    # Verify provenance before replacing any existing figures.
    digest = hashlib.sha256(np.sort(tr).astype("<i8").tobytes()).hexdigest()
    manifest = args.output.parent / "resultados" / "manifiesto_split.json"
    if manifest.exists():
        import json
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if digest != prior["train_index_sha256"] or sha256(args.input) != prior["source_sha256"]:
            raise ValueError("El atlas no coincide con el split o la fuente del EDA anterior")
    args.output.mkdir(parents=True, exist_ok=True)
    figs = args.output / "figuras"
    figs.mkdir(exist_ok=True)
    theme()
    sections = {
        "Numéricas de estudios previos e ingreso": [c for c in numeric if c not in FIRST | SECOND],
        "Numéricas del primer semestre": [c for c in numeric if c in FIRST],
        "Numéricas del segundo semestre": [c for c in numeric if c in SECOND],
        "Categóricas de contexto": [c for c in categoric if c not in {
            "Tiene_deuda", "Pago_de_matrícula_al_día", "Poseedor_de_Beca"}
            and c not in ORDERED],
        "Categóricas administrativas": [c for c in categoric if c in {
            "Tiene_deuda", "Pago_de_matrícula_al_día", "Poseedor_de_Beca"}],
        "Categóricas ordinales familiares": [c for c in categoric if c in ORDERED],
    }
    assert sum(map(len, sections.values())) == 27
    header = [
        "# Atlas de las 27 variables: preguntas, respuestas y gráficos",
        "",
        "Para una lectura guiada y preguntas por cada variable, ver el "
        "[EDA comentado](../EDA_27_VARIABLES_COMENTADO.md).",
        "",
        f"**Base:** 40.000 observaciones de train. Test reservado: {fmt_n(len(test))}. "
        f"Fuente SHA-256 `{sha256(args.input)}`.",
        "",
        "Este atlas se genera con `atlas_variables.py` usando el mismo split estratificado "
        "que `eda_train.py` (20 % test, semilla 42). Ningún gráfico usa el target de test. "
        "Los números son asociaciones descriptivas exploratorias de esta base, no efectos causales.",
        "",
        "## Cómo leer cada figura",
        "",
        "- **Numérica:** izquierda = distribución total de train; derecha = mediana (punto) "
        "y 25.º–75.º percentil por clase. En las ocho académicas, el panel derecho "
        "muestra sólo valores positivos y el texto informa por separado el porcentaje de ceros.",
        "- **Categórica:** izquierda = número de estudiantes por nivel; derecha = porcentaje "
        "Desertor dentro de ese nivel. La línea gris marca el 33,1 % global de train; "
        "el rojo destaca niveles por encima con n≥100. No confundir "
        "P(Desertor | nivel) con P(nivel | Desertor).",
        "- `n` es el denominador de cada categoría. Para todas las tasas y los intervalos de "
        "Wilson por nivel, ver `../resultados/04_categoricas_por_clase_train.csv`. "
        "Para medianas y `n_positivos`, ver `../resultados/03_numericas_por_clase_train.csv`.",
        "",
        "## Antes de interpretar: ¿qué contiene la base?",
        "",
        "El target distingue **Desertor (13.223; 33,1 %), En Curso (7.811; 19,5 %) "
        "y Graduado (18.966; 47,4 %)**. Hay 11 variables numéricas, 16 categóricas "
        "y ningún `NaN` codificado como faltante. Una nota cero puede tener un significado "
        "institucional distinto de una evaluación rendida con nota cero. "
        "[Ver gráfico de clases](../resultados/01_clases_train.png).",
        "",
    ]
    lines = header[:]
    chart_count = 0
    for title, cols in sections.items():
        lines.extend([f"## {title} ({len(cols)})", ""])
        for col in cols:
            chart_count += 1
            slug = f"{chart_count:02d}.svg"
            path = figs / slug
            stats = (numeric_figure(df, col, path) if col in numeric
                     else categorical_figure(df, col, path))
            question = ("¿Qué valores toma, cuántos ceros hay y cómo se distribuyen "
                        "en cada clase?" if col in FIRST | SECOND else
                        "¿Qué valores toma y se solapan entre Desertor, En Curso y Graduado?"
                        if col in numeric else
                        "¿Cuántos estudiantes hay en cada nivel y qué tasa Desertor tiene cada uno?")
            answer = (explain_numeric(col, stats) if col in numeric else explain_category(stats))
            lines.extend([f"### {chart_count:02d}. {PREFIX[col]}", "",
                          f"**Columna:** `{col}` · **Tipo:** "
                          f"{'numérica' if col in numeric else 'categórica ordinal' if col in ORDERED else 'categórica'} "
                          f"· **Momento:** `{phase(col)}`.", "",
                          f"**Pregunta básica:** {question}", "",
                          f"**Respuesta con train:** {answer}", "",
                          f"![{PREFIX[col]}](figuras/{slug})", ""])
    lines.extend([
        "## ¿Cómo paso de estos cruces a variables para el modelo?", "",
        "1. **Revisar validez y momento:** confirmar significado de cero, fecha de captura "
        "y frecuencia de categorías. Una columna muy asociada con target puede ser tardía.",
        "2. **Formular candidatas por EDA:** comparar tasas y distribuciones con `n` "
        "suficiente; consultar las tablas completas antes de fijarse sólo en diferencias "
        "llamativas. `../resultados/05_asociaciones_marginales_train.csv` resume η² "
        "numérica y V de Cramér categórica: **no comparar esas dos escalas entre sí**.",
        "3. **Controlar redundancia y composición:** consultar "
        "`../resultados/06_redundancia_train.csv` y tasas estratificadas. "
        "Una diferencia simple puede desaparecer al considerar carrera u otras columnas.",
        "4. **Elegir la pregunta de modelado:** usar **27 predictores juntos** para la "
        "comparación general del TP, o comparar conjuntos **12 candidatas al ingreso → "
        "+4 de S1 → +4 de S2**, y evaluar las **7 sin fecha** en sensibilidad. "
        "Esos grupos proceden del significado/fecha sugerida de las columnas, "
        "**no de un ranking elegido por el target**.",
        "5. **Medir aporte conjunto:** comparar score de validación con las mismas filas, "
        "folds, familia de modelo y preprocesamiento. Una asociación bivariada "
        "no asegura mejora en presencia de otras variables. Reservar test; "
        "este conjunto ya se vio en el TP previo, por lo que las comparaciones "
        "son exploratorias y necesitan otra cohorte para confirmación externa.",
        "",
        "**Límite temporal:** no se dispone de fecha individual de abandono ni "
        "de medición verificada para las siete variables apartadas. Incorporar S1 o S2 "
        "mide información adicional de esas columnas en la base, pero no prueba "
        "detección anterior a la deserción.", "",
    ])
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Atlas generado: {chart_count} variables, {len(tr)} train, {len(test)} test reservado")


if __name__ == "__main__":
    main()
