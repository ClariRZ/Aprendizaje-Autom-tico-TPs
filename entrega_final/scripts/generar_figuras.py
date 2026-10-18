"""Genera las siete figuras del informe con un estilo común de artículo.

Desde la raíz del repositorio:
    python 00_base_de_datos/reconstruir_dataset.py
    python entrega_final/scripts/predicciones_test.py   # una vez; guarda salidas de test
    python entrega_final/scripts/generar_figuras.py

Convenciones (formato de artículo tipo MDPI): sin títulos dentro de la figura,
paneles identificados (a), (b)… debajo de cada eje y descripción completa en
el epígrafe del informe; coma decimal; paleta sobria.

Fuentes:
- Figura 1 (EDA): recalculada sobre el entrenamiento de la partición multiclase.
- Figuras 2 y 3 (test): datos_figuras/, generados por predicciones_test.py.
- Figura 4: puntos de los SVG originales del bloque 04 (curvas binarias).
- Figuras 5 y 6: CSV de AA privada/03_verificaciones_catedra y 02_alternativa_temporal.
- Figura 7: datos_figuras/importancia_permutacion.csv (informe de importancia).
No entrena modelos ni elige configuraciones.
"""
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "entrega_final"
OUT, DATOS = BASE / "figuras", BASE / "datos_figuras"
PRIV = ROOT / "AA privada"
sys.path.insert(0, str(ROOT / "Resolución TP AA"))
from _comun import cargar_eda_train  # noqa: E402

# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------
INK, MUTED, GRID = "#262626", "#6e6e6e", "#e3e3e3"
TRAIN, VALID, ACENTO = "#a6a6a6", "#2f4f6f", "#9c3d3d"
CLASES = {"Desertor": "#9c3d3d", "En Curso": "#c08a3e", "Graduado": "#3f5f7f"}
ETAPAS = ["#9a9a9a", "#2f4f6f", "#7f9bb5", "#6b7f5a"]
SECUENCIAL = LinearSegmentedColormap.from_list("pizarra", ["#ffffff", "#c9d5e1", "#2f4f6f", "#1c3147"])
ANCHO = 6.3  # pulgadas: ancho de la columna de texto del informe

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.labelcolor": INK, "axes.edgecolor": "#b5b5b5", "axes.linewidth": .7,
    "text.color": INK, "xtick.color": "#4d4d4d", "ytick.color": "#4d4d4d",
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "xtick.major.size": 2.5,
    "ytick.major.size": 0, "legend.fontsize": 7.5, "legend.frameon": False,
    "svg.fonttype": "path", "savefig.facecolor": "white", "figure.dpi": 150,
    "mathtext.default": "regular",
})


def coma(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",")


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def eje_coma(ax, dec=2, eje="y"):
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(FuncFormatter(lambda v, _: coma(v, dec)))


def limpio(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.6)
    ax.set_axisbelow(True)


def panel(ax, letra, xlabel=""):
    """Etiqueta del eje x seguida del identificador del panel, como en MDPI."""
    ax.set_xlabel((xlabel + "\n" if xlabel else "") + r"$\mathbf{(" + letra + r")}$", linespacing=1.6)


def rotulo(ax, texto, x=.98, ha="right"):
    """Rótulo breve dentro del panel (arriba)."""
    ax.text(x, .97, texto, transform=ax.transAxes, ha=ha, va="top", fontsize=7.5, color=INK,
            bbox=dict(boxstyle="square,pad=.25", fc="white", ec="none", alpha=.85))


def guardar(fig, nombre):
    fig.savefig(OUT / nombre, bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / nombre.replace(".svg", ".pdf"), bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    print("  ", nombre)


# ---------------------------------------------------------------------------
# Figura 1. Asociaciones exploratorias en entrenamiento
# ---------------------------------------------------------------------------
def fig_eda():
    tr = cargar_eda_train()
    aprob = ["Unidades_curriculares_1er_semestre_(aprobadas)", "Unidades_curriculares_2do_semestre_(aprobadas)"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(ANCHO, 2.7), gridspec_kw={"width_ratios": [1, 1.1]})

    x = np.array([0, 1])
    w = .26
    for k, c in enumerate(CLASES):
        sub = tr[tr.target == c]
        vals = [100 * (sub[col] == 0).mean() for col in aprob]
        pos = x + (k - 1) * w
        a.bar(pos, vals, width=w * .92, color=CLASES[c], label=f"{c} (n = {miles(len(sub))})")
        for p, v in zip(pos, vals):
            a.text(p, v + 1.5, coma(v, 1), ha="center", fontsize=7)
    a.set_xticks(x, ["Semestre 1", "Semestre 2"])
    a.tick_params(axis="x", length=0)
    a.set_ylim(0, 80)
    a.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    a.set_ylabel("Registros sin materias aprobadas")
    limpio(a)
    panel(a, "a")
    fig.legend(*a.get_legend_handles_labels(), loc="upper left", ncol=3, handlelength=1,
               bbox_to_anchor=(.02, 1.07))

    filas = [("Matrícula no al día", tr["Pago_de_matrícula_al_día"] == "No"),
             ("Matrícula al día", tr["Pago_de_matrícula_al_día"] == "Sí"),
             ("Con deuda", tr["Tiene_deuda"] == "Sí"),
             ("Sin deuda", tr["Tiene_deuda"] == "No"),
             ("Sin beca", tr["Poseedor_de_Beca"] == "No"),
             ("Con beca", tr["Poseedor_de_Beca"] == "Sí")]
    total = 100 * (tr.target == "Desertor").mean()
    pos = np.array([0, 1, 2.4, 3.4, 4.8, 5.8])
    for p, (lab, m) in zip(pos, filas):
        v = 100 * (tr.loc[m, "target"] == "Desertor").mean()
        b.barh(p, 100, height=.62, color="#ececec")
        b.barh(p, v, height=.62, color=CLASES["Desertor"])
        dentro = v > 85
        b.text(v - 1.5 if dentro else v + 1.5, p, coma(v, 1) + " %", va="center",
               ha="right" if dentro else "left", fontsize=7, color="white" if dentro else INK)
    b.axvline(total, color=INK, lw=.8, ls=(0, (3, 2)))
    b.text(total + 1, -.75, f"Total: {coma(total, 1)} %", fontsize=7, color=INK, va="center")
    b.set_yticks(pos, [f"{lab} (n = {miles(m.sum())})" for lab, m in filas], fontsize=7)
    b.set_ylim(6.3, -1.1)
    b.set_xlim(0, 100)
    b.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    b.spines[["top", "right", "left"]].set_visible(False)
    panel(b, "b", "Proporción de Desertor dentro del grupo")
    fig.subplots_adjust(wspace=.72)
    guardar(fig, "figura_01.svg")


# ---------------------------------------------------------------------------
# Figura 2. Curvas ROC binarias en prueba
# ---------------------------------------------------------------------------
MODELOS_BIN = {"Logistica": ("Logística", "#2f4f6f", "-"), "SVM": ("SVM", "#9a9a9a", "-"),
               "Arbol": ("Árbol", "#c08a3e", "-"), "Soft Voting": ("Soft Voting", INK, (0, (4, 2)))}


def fig_roc():
    d = pd.read_csv(DATOS / "puntajes_binaria.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(ANCHO, 3.0))
    for col, (nom, color, ls) in MODELOS_BIN.items():
        fpr, tpr, _ = roc_curve(d.y, d[col])
        auc = roc_auc_score(d.y, d[col])
        pred = (d[col] > 0) if col == "SVM" else (d[col] >= .5)
        sens = (pred & (d.y == 1)).sum() / (d.y == 1).sum()
        fp = (pred & (d.y == 0)).sum() / (d.y == 0).sum()
        for ax in (a, b):
            ax.plot(fpr, tpr, color=color, ls=ls, lw=1.2, label=f"{nom} (AUC = {coma(auc, 4)})")
        b.scatter(fp, sens, s=24, color=color, edgecolor="white", lw=.7, zorder=4)
    a.plot([0, 1], [0, 1], color="#c8c8c8", lw=.7, ls=(0, (3, 2)))
    a.add_patch(plt.Rectangle((0, .6), .2, .4, fill=False, ec=MUTED, lw=.6, ls=(0, (2, 2))))
    a.set(xlim=(0, 1), ylim=(0, 1.01))
    a.set_ylabel("Sensibilidad")
    a.legend(loc="lower right", handlelength=1.8)
    b.set(xlim=(0, .2), ylim=(.6, 1.0))
    rotulo(b, "Puntos: corte por defecto", x=.97)
    for ax, dec in ((a, 1), (b, 2)):
        eje_coma(ax, dec)
        eje_coma(ax, dec, "x")
        limpio(ax, "both")
    panel(a, "a", "1 − especificidad")
    panel(b, "b", "1 − especificidad")
    fig.subplots_adjust(wspace=.25)
    guardar(fig, "figura_02.svg")


# ---------------------------------------------------------------------------
# Figura 3. Matrices de confusión de los cuatro modelos multiclase
# ---------------------------------------------------------------------------
MODELOS_MC = {"KNN": "K-NN", "Arbol_multiclase": "Árbol", "RandomForest": "Random Forest",
              "AdaBoost": "AdaBoost"}


def fig_matrices():
    d = pd.read_csv(DATOS / "predicciones_multiclase.csv")
    clases = list(CLASES)
    fig, axes = plt.subplots(1, 4, figsize=(ANCHO + .5, 2.15))
    for k, (ax, (col, nom)) in enumerate(zip(axes, MODELOS_MC.items())):
        cm = confusion_matrix(d.y, d[col], labels=clases)
        pct = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(pct, cmap=SECUENCIAL, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100 * pct[i, j]:.0f} %", ha="center", va="center", fontsize=7,
                        color="white" if pct[i, j] > .5 else INK)
        ax.set_xticks(range(3), ["Des.", "En C.", "Grad."])
        ax.set_yticks(range(3), ["Desertor", "En Curso", "Graduado"] if k == 0 else [])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(nom, fontsize=7.5, pad=4)
        panel(ax, "abcd"[k], "Clase predicha")
    axes[0].set_ylabel("Clase real")
    cax = fig.add_axes([.925, .3, .012, .55])
    cb = fig.colorbar(im, cax=cax, ticks=[0, .25, .5, .75, 1])
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=0)
    fig.subplots_adjust(wspace=.12, right=.91)
    guardar(fig, "figura_03.svg")


# ---------------------------------------------------------------------------
# Figuras 4 y 5. Curvas de complejidad
# ---------------------------------------------------------------------------
def svg_original(nombre):
    """Recupera los puntos de las curvas SVG publicadas por el bloque 04."""
    svg = (ROOT / "Resolución TP AA/informe complejidad y generalización/graficos" / nombre).read_text(encoding="utf-8")
    ticks = re.findall(r'<line x1="90" y1="([0-9.]+)" x2="950" y2="[0-9.]+" class="grid"/>'
                       r'<text x="78" y="[0-9.]+" text-anchor="end" class="small">([0-9.]+)</text>', svg)
    lineas = re.findall(r'<polyline points="([^"]+)"', svg)
    (y0, v0), (y1, v1) = ((float(a), float(b)) for a, b in ticks[:2])
    return [np.array([v0 + (float(p.split(",")[1]) - y0) * (v1 - v0) / (y1 - y0) for p in pts.split()])
            for pts in lineas]


def curva(ax, x, tr, va, sd=None, elegido=None):
    ax.plot(x, tr, color=TRAIN, lw=1.3, ls=(0, (4, 2)), marker="o", ms=2.8, label="Entrenamiento")
    if sd is not None:
        ax.fill_between(x, va - sd, va + sd, color=VALID, alpha=.13, lw=0)
    ax.plot(x, va, color=VALID, lw=1.5, marker="o", ms=3.2, label="Validación")
    if elegido is not None:
        ax.scatter(x[elegido], va[elegido], s=80, facecolor="none", edgecolor=ACENTO, lw=1.2, zorder=5)


def leyenda_curvas(fig, ax, banda=False, y=-.06):
    h, l = ax.get_legend_handles_labels()
    if banda:
        h.append(Patch(color=VALID, alpha=.13))
        l.append("± 1 DE entre folds")
    h.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec=ACENTO, ms=7))
    l.append("Configuración final")
    fig.legend(h, l, loc="lower center", ncol=len(h), bbox_to_anchor=(.5, y))


def fig_complejidad_binaria():
    specs = [("01_logistica_C.svg", "Logística", "C", ["0,01", "0,1", "1", "10"], 1),
             ("02_svm_C.svg", "SVM (γ = 0,01)", "C", ["0,1", "1", "10"], 0),
             ("04_arbol_depth.svg", "Árbol (hoja mín. 50)", "Profundidad máxima",
              ["1", "2", "3", "4", "6", "8", "12", "20", "∞"], 5)]
    fig, axes = plt.subplots(1, 3, figsize=(ANCHO, 2.3))
    for k, (ax, (nom, tit, xl, labs, eleg)) in enumerate(zip(axes, specs)):
        tr, va = svg_original(nom)
        x = np.arange(len(labs))
        curva(ax, x, tr, va, elegido=eleg)
        ax.set_xticks(x, labs)
        lo, hi = min(tr.min(), va.min()), max(tr.max(), va.max())
        pad = max(.004, (hi - lo) * .18)
        ax.set_ylim(lo - pad, hi + pad * 1.8)
        ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5]))
        eje_coma(ax, 3)
        limpio(ax)
        rotulo(ax, tit)
        panel(ax, "abc"[k], xl)
    axes[0].set_ylabel("ROC-AUC media (3 folds)")
    leyenda_curvas(fig, axes[0], y=-.12)
    fig.subplots_adjust(wspace=.38, bottom=.27)
    guardar(fig, "figura_04.svg")


def fig_complejidad_multiclase():
    folds = pd.read_csv(PRIV / "03_verificaciones_catedra/resultados/scores_folds.csv")
    ada = pd.read_csv(PRIV / "03_verificaciones_catedra/resultados/extension_ada_curva.csv")
    ada = ada.rename(columns={"f1_macro_ajuste": "train", "f1_macro_validacion": "validacion"})
    specs = [("Arbol", "Árbol (hoja mín. 50) · n = 40.000", "Profundidad máxima", ["4", "8", "12", "sin_limite"], 2),
             ("RandomForest", "RF (80 árboles, hoja mín. 10) · n = 12.000", "Profundidad máxima",
              ["4", "8", "12", "sin_limite"], None),
             ("KNN", "K-NN · n = 6.000", "Número de vecinos", ["3", "11", "31", "101"], None),
             ("AdaBoost", "AdaBoost (prof. 2, tasa 0,1) · n = 12.000", "Número de árboles",
              ["100", "200", "400", "600"], 2)]
    fig, axes = plt.subplots(2, 2, figsize=(ANCHO, 4.4))
    for k, (ax, (mod, tit, xl, orden, eleg)) in enumerate(zip(axes.flat, specs)):
        g = ada if mod == "AdaBoost" else folds[(folds.seccion == "curva_complejidad") & (folds.modelo == mod)]
        g = g.assign(valor=g.valor.astype(str))
        m = g.groupby("valor")[["train", "validacion"]].mean().reindex(orden)
        s = g.groupby("valor")["validacion"].std(ddof=0).reindex(orden)
        x = np.arange(len(orden))
        curva(ax, x, m.train.to_numpy(), m.validacion.to_numpy(), s.to_numpy(), eleg)
        best = int(np.argmax(m.validacion))
        ax.annotate(coma(m.validacion.iloc[best], 3), (x[best], m.validacion.iloc[best]),
                    xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7, color=VALID)
        ax.set_xticks(x, ["Sin límite" if v == "sin_limite" else v for v in orden])
        ax.set_ylim(.68, .86)
        ax.yaxis.set_major_locator(MultipleLocator(.04))
        eje_coma(ax, 2)
        limpio(ax)
        ax.set_title(tit, loc="left", fontsize=7.5, pad=4)
        panel(ax, "abcd"[k], xl)
        if k % 2 == 0:
            ax.set_ylabel("F1 macro media (3 folds)")
    leyenda_curvas(fig, axes[0, 0], banda=True, y=-.07)
    fig.subplots_adjust(wspace=.22, hspace=.62, bottom=.13)
    guardar(fig, "figura_05.svg")


# ---------------------------------------------------------------------------
# Figura 6. Análisis acumulativo con referencia Dummy
# ---------------------------------------------------------------------------
def fig_acumulativo():
    met = pd.read_csv(PRIV / "02_alternativa_temporal/resultados_modelos/metricas.csv")
    met = met[met.subconjunto == "test_completo"]
    etapas = ["E0_entrada_candidata", "E1_mas_semestre1", "E2_mas_semestre2", "E2_con_sin_fecha"]
    labs = ["Inscripción\n(12)", "+ S1\n(16)", "+ S2\n(20)", "+ Admin.\n(27)"]
    specs = [("binaria", "ROC-AUC media (3 folds)", .5, "Binaria · regresión logística", (.45, 1.0)),
             ("multiclase", "F1 macro media (3 folds)", None, "Multiclase · árbol", (.15, .85))]
    fig, axes = plt.subplots(1, 2, figsize=(ANCHO, 2.6))
    for k, (ax, (tarea, yl, dummy, tit, ylim)) in enumerate(zip(axes, specs)):
        m = met[met.tarea == tarea].set_index("etapa")
        v = m.loc[etapas, "cv_media"].to_numpy()
        if dummy is None:
            dummy = m.loc["Dummy_mas_frecuente", "f1_macro"]
        x = np.arange(4)
        ax.plot(x[:3], v[:3], color=VALID, lw=1.5, zorder=2)
        ax.plot(x[2:], v[2:], color=VALID, lw=1.5, ls=(0, (3, 2)), zorder=2)
        ax.scatter(x, v, s=36, color=ETAPAS, zorder=3, edgecolor="white", lw=.8)
        alto = ylim[1] - ylim[0]
        for i in range(4):
            ax.text(x[i], v[i] + .045 * alto, coma(v[i], 3), ha="center", fontsize=7)
            if i:
                ax.text((x[i] + x[i - 1]) / 2, ylim[0] + .2 * alto, "+" + coma(v[i] - v[i - 1], 3),
                        ha="center", fontsize=7, color=INK if i == 1 else MUTED,
                        fontweight="bold" if i == 1 else "normal")
        ax.text(-.38, ylim[0] + .27 * alto, "Cambio entre etapas", fontsize=6.5, color=MUTED)
        ax.axhline(dummy, color=MUTED, lw=.7, ls=(0, (1, 2)))
        ax.text(3.38, dummy + .015 * alto, f"Dummy: {coma(dummy, 3)}", ha="right", fontsize=6.5, color=MUTED)
        ax.set_xticks(x, labs)
        ax.set_xlim(-.4, 3.4)
        ax.set_ylim(*ylim)
        ax.set_ylabel(yl)
        eje_coma(ax, 1)
        limpio(ax)
        ax.set_title(tit, loc="left", fontsize=7.5, pad=4)
        panel(ax, "ab"[k], "Predictores incluidos")
    fig.subplots_adjust(wspace=.32)
    guardar(fig, "figura_06.svg")


# ---------------------------------------------------------------------------
# Figura 7. Importancia por permutación
# ---------------------------------------------------------------------------
def fig_importancia():
    d = pd.read_csv(DATOS / "importancia_permutacion.csv").iloc[::-1]
    col = {"Académicas": VALID, "Deuda / matrícula / beca": "#c08a3e", "Otras 16 variables": "#a6a6a6"}
    fig, ax = plt.subplots(figsize=(ANCHO, 2.9))
    y = np.arange(len(d))
    ax.barh(y, d.caida_f1_macro, color=d.grupo.map(col), height=.62)
    ax.errorbar(d.caida_f1_macro, y, xerr=d.sd, fmt="none", ecolor=INK, elinewidth=.6, capsize=1.6)
    for yi, v, s in zip(y, d.caida_f1_macro, d.sd):
        ax.text(v + s + .0015, yi, coma(v, 4), va="center", fontsize=7)
    ax.set_yticks(y, d.variable)
    ax.set_xlim(0, .118)
    ax.set_xlabel("Caída media del F1 macro de validación al permutar la variable (± 1 DE)")
    eje_coma(ax, 2, "x")
    limpio(ax, "x")
    ax.legend(handles=[Patch(color=c) for c in col.values()], loc="lower right",
              labels=["Académicas (8 variables; suma 0,1533)", "Deuda, matrícula y beca (suma 0,0468)",
                      "Otras 16 variables (suma 0,0075)"])
    guardar(fig, "figura_07.svg")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("figura_*"):
        f.unlink()
    print("Figuras:")
    fig_eda()
    fig_roc()
    fig_matrices()
    fig_complejidad_binaria()
    fig_complejidad_multiclase()
    fig_acumulativo()
    fig_importancia()
