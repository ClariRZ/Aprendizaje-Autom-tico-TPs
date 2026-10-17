"""Figuras autocontenidas del TP, a partir de resultados de validación en train.

Desde la raíz del repositorio: python "Resolución TP AA/tp completo/generar_figuras_ampliacion.py"
Lee 03_verificaciones_catedra/resultados/scores_folds.csv y
extension_ada_curva.csv, y las métricas de 02_alternativa_temporal.
No entrena modelos ni consulta las etiquetas de test para estas curvas.
"""

from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "AA privada/03_verificaciones_catedra/resultados/scores_folds.csv").exists())
FOLDS = ROOT / "AA privada/03_verificaciones_catedra/resultados/scores_folds.csv"
ETAPAS = ROOT / "AA privada/02_alternativa_temporal/resultados_modelos/metricas.csv"
ADA_EXT = ROOT / "AA privada/03_verificaciones_catedra/resultados/extension_ada_curva.csv"
OUT = Path(__file__).resolve().parent / "graficos"
GRAY, LIGHT, RED = "#474747", "#aaaaaa", "#bb343c"


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelcolor": GRAY, "text.color": GRAY, "xtick.color": GRAY,
        "ytick.color": GRAY, "axes.edgecolor": "#dddddd",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=.8)
    ax.set_axisbelow(True)


def read_original_svg(name):
    """Recupera puntos de las curvas SVG ya publicadas por el script original."""
    base = ROOT / "Resolución TP AA/informe complejidad y generalización/graficos"
    svg = (base / name).read_text(encoding="utf-8")
    ticks = re.findall(
        r'<line x1="90" y1="([0-9.]+)" x2="950" y2="[0-9.]+" class="grid"/>'
        r'<text x="78" y="[0-9.]+" text-anchor="end" class="small">([0-9.]+)</text>', svg)
    polylines = re.findall(r'<polyline points="([^"]+)"', svg)
    if len(ticks) < 2 or len(polylines) != 2:
        raise ValueError(f"SVG original inesperado: {name}")
    (y0, v0), (y1, v1) = ((float(y), float(v)) for y, v in ticks[:2])
    values = []
    for points in polylines:
        ys = [float(p.split(",")[1]) for p in points.split()]
        values.append([v0 + (y - y0) * (v1 - v0) / (y1 - y0) for y in ys])
    return values


def binary_complexity():
    specs = [
        ("01_logistica_C.svg", "Logística · C", ["0,01", "0,1", "1", "10"]),
        ("02_svm_C.svg", "SVM · C", ["0,1", "1", "10"]),
        ("04_arbol_depth.svg", "Árbol · profundidad",
         ["1", "2", "3", "4", "6", "8", "12", "20", "∞"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, (name, title, labels) in zip(axes, specs):
        tr, cv = read_original_svg(name)
        if len(tr) != len(labels):
            raise ValueError(name)
        x = np.arange(len(labels))
        ax.plot(x, tr, "o-", color=GRAY, lw=2, markersize=4)
        ax.plot(x, cv, "o-", color=RED, lw=2, markersize=4)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_ylim(max(0, min(tr + cv) - .012), min(1, max(tr + cv) + .012))
        tidy(ax)
    fig.suptitle("Tres modelos binarios: ¿qué cambia al darles más libertad?",
                 x=.03, y=.99, ha="left", fontsize=15, fontweight="bold")
    fig.text(.03, .025, "Gris: entrenamiento · Rojo: validación. AUC media de tres folds en 40.000 filas; cada panel tiene distinta escala vertical.",
             fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, bottom=.2, top=.78, wspace=.31)
    fig.savefig(OUT / "04_complejidad_binaria_resumen.svg", bbox_inches="tight")
    plt.close(fig)


def complexity(data, ada_extended):
    names = [
        ("Arbol", "Árbol · profundidad", ["4", "8", "12", "sin_limite"], "40.000"),
        ("RandomForest", "Bosque · profundidad", ["4", "8", "12", "sin_limite"], "12.000"),
        ("KNN", "K-NN · vecinos", ["3", "11", "31", "101"], "6.000"),
        ("AdaBoost", "AdaBoost · árboles (prof. 2)", ["100", "200", "400", "600"], "12.000"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.1), constrained_layout=False)
    for ax, (name, title, order, n) in zip(axes.flat, names):
        if name == "AdaBoost":
            g = ada_extended.copy()
            g["valor"] = g["valor"].astype(str)
            v = g.groupby("valor")[["f1_macro_ajuste", "f1_macro_validacion"]].mean()
            v = v.rename(columns={"f1_macro_ajuste": "train", "f1_macro_validacion": "validacion"}).reindex(order)
        else:
            g = data[(data.seccion == "curva_complejidad") & (data.modelo == name)]
            v = g.groupby("valor")[["train", "validacion"]].mean().reindex(order)
        if v.isna().any().any():
            raise ValueError(f"Faltan puntos en {name}")
        x = np.arange(len(v))
        ax.plot(x, v.train, "o-", color=GRAY, lw=2, markersize=5)
        ax.plot(x, v.validacion, "o-", color=RED, lw=2, markersize=5)
        ax.set_xticks(x, ["Sin límite" if q == "sin_limite" else q for q in order])
        lo, hi = min(v.min()) - .02, max(v.max()) + .025
        ax.set_ylim(max(0, lo), min(1, hi))
        ax.set_title(title, loc="left", fontweight="bold", y=1.13)
        ax.text(0, 1.045, f"F1 macro · {n} filas de train", transform=ax.transAxes,
                va="bottom", fontsize=9, color=GRAY)
        ax.text(x[np.argmax(v.validacion.to_numpy())],
                max(v.validacion) + .006, f"{max(v.validacion):.3f}",
                color=RED, fontsize=9, ha="center")
        tidy(ax)
    fig.suptitle("¿Cuándo deja de ayudar la complejidad?", fontsize=16,
                 fontweight="bold", x=.07, ha="left", y=.985)
    fig.text(.07, .925, "Gris: ajuste en entrenamiento  ·  Rojo: evaluación en folds no vistos",
             fontsize=10)
    fig.text(.07, .038, "Cada punto: media de 3 folds. Escala vertical y tamaño de muestra distintos por panel; comparar la forma de cada curva.",
             fontsize=9, color=GRAY)
    fig.text(.07, .017, "En K-NN, más vecinos simplifica; en AdaBoost se fijaron profundidad 2 y tasa 0,1. Curvas diagnósticas, sin uso de test.",
             fontsize=9, color=GRAY)
    fig.subplots_adjust(left=.08, right=.97, bottom=.11, top=.83, wspace=.27, hspace=.43)
    fig.savefig(OUT / "08_complejidad_multiclase.svg", bbox_inches="tight")
    plt.close(fig)


def baseline(data):
    d = data[data.seccion == "sensibilidad_folds"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, task, name1, name2, title, metric in [
        (axes[0], "binaria", "Dummy_binario", "Logistica", "Desertor frente a los demás", "ROC-AUC"),
        (axes[1], "multiclase", "Dummy_multiclase", "Arbol", "Tres estados académicos", "F1 macro"),
    ]:
        for j, name in enumerate([name1, name2]):
            for folds in (3, 5):
                g = d[(d.modelo == name) & (d.folds == folds) & (d.tarea == task)]
                if len(g) != folds:
                    raise ValueError((name, folds, len(g)))
                value = g.validacion.mean()
                y = j * 1.0 + (folds - 3) * .12
                ax.plot([0, value], [y, y], color="#e6e6e6", lw=2)
                ax.plot(value, y, "o", ms=8, color=GRAY if folds == 3 else RED)
                ax.text(value + .018, y, f"{value:.3f}", va="center", fontsize=9,
                        color=GRAY if folds == 3 else RED)
        ax.set_yticks([.12, 1.12], ["Predicción más frecuente", "Logística" if task == "binaria" else "Árbol"])
        ax.set_xlim(0, 1.07)
        ax.set_ylim(-.3, 1.53)
        ax.set_xlabel(metric + " en validación")
        ax.set_title(title, loc="left", fontweight="bold")
        tidy(ax)
    fig.suptitle("Los modelos superan una regla que ignora las variables",
                 x=.04, y=.99, ha="left", fontsize=15, fontweight="bold")
    fig.text(.04, .005, "Gris: 3 folds  ·  Rojo: 5 folds. Comparar dentro de cada tarea: las métricas son distintas.", fontsize=9)
    fig.subplots_adjust(left=.21, right=.96, bottom=.21, top=.79, wspace=.35)
    fig.savefig(OUT / "09_dummy_y_folds.svg", bbox_inches="tight")
    plt.close(fig)


def stages(metrics):
    d = metrics[(metrics.subconjunto == "test_completo") &
                metrics.etapa.isin(["E0_entrada_candidata", "E1_mas_semestre1",
                                  "E2_mas_semestre2", "E2_con_sin_fecha"])].copy()
    order = ["E0_entrada_candidata", "E1_mas_semestre1", "E2_mas_semestre2", "E2_con_sin_fecha"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, task, score, title in [
        (axes[0], "binaria", "cv_media", "Binaria · ROC-AUC en CV"),
        (axes[1], "multiclase", "cv_media", "Tres clases · F1 macro en CV"),
    ]:
        v = d[d.tarea == task].set_index("etapa").reindex(order)
        if v[score].isna().any():
            raise ValueError((task, v[score]))
        yy = v[score].to_numpy()
        ax.plot(range(3), yy[:3], "o-", color=GRAY, lw=2.5, markersize=8)
        ax.plot([2, 3], yy[2:], "--", color=LIGHT, lw=1.7)
        ax.plot(3, yy[3], "D", color=RED, markersize=8)
        for x, y in enumerate(yy):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9, color=RED if x == 3 else GRAY)
        ax.set_xticks(range(4), ["Ingreso\n12", "+ semestre 1\n16", "+ semestre 2\n20", "+ otras 7\n27"])
        ax.set_xlim(-.23, 3.23)
        ax.set_ylim(0, 1.04)
        ax.set_title(title, loc="left", fontweight="bold")
        tidy(ax)
    fig.suptitle("Desempeño al acumular grupos de variables",
                 x=.04, y=.99, ha="left", fontsize=15, fontweight="bold")
    fig.text(.04, .01, "Mismos casos y folds por etapa; selección de parámetros en train. Rojo: siete variables administrativas y familiares.", fontsize=9)
    fig.subplots_adjust(left=.075, right=.96, bottom=.26, top=.78, wspace=.26)
    fig.savefig(OUT / "10_senales_por_etapa.svg", bbox_inches="tight")
    plt.close(fig)


def confusion_forest():
    """Matriz del bosque elegido: porcentaje de cada clase real en test."""
    d = pd.read_csv(Path(__file__).with_name("matriz_rf.csv"))
    labels = ["Desertor", "En Curso", "Graduado"]
    counts = d.pivot(index="real", columns="predicha", values="n").loc[labels, labels]
    pct = d.pivot(index="real", columns="predicha", values="porcentaje_fila").loc[labels, labels]
    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(2.5, -.5)
    for row in range(3):
        for col in range(3):
            val = pct.iloc[row, col]
            face = "#f2f2f2" if row == col else ("#f4dee0" if val >= 10 else "#fafafa")
            ax.add_patch(plt.Rectangle((col-.47, row-.43), .94, .86,
                                       facecolor=face, edgecolor="white"))
            ax.text(col, row-.06, f"{val:.1f} %".replace(".", ","),
                    ha="center", va="center", color=RED if row != col else GRAY,
                    fontsize=17, fontweight="bold")
            ax.text(col, row+.22, f"{counts.iloc[row, col]:,}".replace(",", ".")+" casos",
                    ha="center", va="center", color=GRAY, fontsize=9)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), [f"{name} · n={sum(counts.loc[name]):,}".replace(",", ".")
                               for name in labels])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.07, .99, "¿Qué estado predice el bosque para cada estado real?",
             fontsize=14, fontweight="bold", color=GRAY, ha="left", va="top")
    fig.text(.07, .88, "Columnas: predicción   ·   Filas: estado registrado en test",
             fontsize=9, color=GRAY, ha="left")
    fig.text(.07, .015, "Cada fila suma 100 %. Rojo: errores; los conteos provienen de 10.000 casos de test.",
             fontsize=8.5, color=GRAY, ha="left")
    fig.subplots_adjust(left=.30, right=.97, bottom=.09, top=.70)
    fig.savefig(OUT / "11_matriz_rf.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    setup()
    data, metrics = pd.read_csv(FOLDS), pd.read_csv(ETAPAS)
    assert len(data) == 130 and metrics.etapa.nunique() >= 6
    binary_complexity()
    complexity(data, pd.read_csv(ADA_EXT))
    baseline(data)
    stages(metrics)
    confusion_forest()
    print("Creadas:", *(str(OUT / n) for n in (
        "04_complejidad_binaria_resumen.svg", "08_complejidad_multiclase.svg",
        "09_dummy_y_folds.svg", "10_senales_por_etapa.svg", "11_matriz_rf.svg")), sep="\n")
