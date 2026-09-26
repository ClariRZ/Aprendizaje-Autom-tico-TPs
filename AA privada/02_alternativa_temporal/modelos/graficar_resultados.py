"""Figura de comparación por etapas: solo scores de CV en train."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / "resultados_modelos"
FOLDS = pd.read_csv(BASE / "validacion_folds.csv")
ETAPAS = ["E0", "E1", "E2"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
for ax, tarea, titulo, metrica in zip(
    axes,
    ["binaria", "multiclase"],
    ["Desertor vs resto", "Tres trayectorias"],
    ["ROC-AUC", "F1 macro"],
):
    d = FOLDS[FOLDS.tarea == tarea]
    for con_fecha, color, estilo, leyenda in [
        (False, "#A92F36", "-", "Sin variables de fecha incierta"),
        (True, "#777777", "--", "Incluye 7 variables sin fecha"),
    ]:
        etapa_names = (["E0_con_sin_fecha", "E1_con_sin_fecha", "E2_con_sin_fecha"]
                       if con_fecha else
                       ["E0_entrada_candidata", "E1_mas_semestre1", "E2_mas_semestre2"])
        m = [d.loc[d.etapa == e, "score_validacion"].mean() for e in etapa_names]
        lo = [d.loc[d.etapa == e, "score_validacion"].min() for e in etapa_names]
        hi = [d.loc[d.etapa == e, "score_validacion"].max() for e in etapa_names]
        ax.plot(ETAPAS, m, linestyle=estilo, marker="o", color=color,
                linewidth=2, label=leyenda)
        ax.fill_between(ETAPAS, lo, hi, color=color, alpha=.12)
        for i, value in enumerate(m):
            ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 9 if con_fecha else -16),
                        textcoords="offset points", ha="center", fontsize=8, color=color)
    ax.set(title=titulo, ylabel=metrica, xlabel="Información disponible (etapas candidatas)")
    ax.grid(axis="y", color="#E1E1E1", linewidth=.8)
    ax.set_ylim((.73, .98) if tarea == "binaria" else (.50, .82))
axes[0].legend(loc="lower right", fontsize=8)
fig.suptitle("Validación cruzada en train, mismas filas y folds entre etapas", fontsize=13)
fig.text(.5, .005, "Banda: rango de 3 folds. E0 y el momento de captura de 7 columnas requieren verificación.",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0, .045, 1, .94))
fig.savefig(BASE / "validacion_por_etapa.png", dpi=160)
plt.close(fig)
