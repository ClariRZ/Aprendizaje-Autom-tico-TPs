"""Dos figuras de EDA para el informe; calcula todo exclusivamente en train.

Ejecutar desde cualquier directorio con:
    python "Resolución TP AA/tp completo/generar_figuras_eda_sintesis.py"

Requiere que la base reconstruida esté en 00_base_de_datos/dataset_TP1_2026.csv.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

plt.rcParams["svg.fonttype"] = "none"  # Texto seleccionable y SVG livianos.


BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "00_base_de_datos" / "dataset_TP1_2026.csv"
OUT = Path(__file__).resolve().parent / "graficos"
OUT.mkdir(parents=True, exist_ok=True)

RED = "#C32630"
DARK = "#3E3E3E"
MID = "#838383"
LIGHT = "#C5C5C5"
GRID = "#E7E7E7"


def style(ax, max_x, ticks):
    ax.set_xlim(0, max_x)
    ax.set_xticks(ticks, [f"{x:.0f}%" for x in ticks])
    ax.tick_params(axis="both", which="both", length=0, pad=8, labelsize=10, colors=DARK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    df = pd.read_csv(DATA)
    train_index, _ = train_test_split(
        np.arange(len(df)), test_size=0.20, stratify=df["target"], random_state=42
    )
    train = df.iloc[train_index].copy()
    assert len(train) == 40000
    groups = ["Desertor", "En Curso", "Graduado"]
    sizes = train["target"].value_counts()
    palette = [RED, MID, LIGHT]

    # Pregunta: ¿cuánto de la separación académica procede de cero aprobadas?
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True)
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.94, "El cero en aprobadas distingue las trayectorias", fontsize=16,
             weight="bold", color=DARK)
    fig.text(0.05, 0.865, "Porcentaje con cero materias aprobadas dentro de cada clase · train, n = 40.000",
             fontsize=10.5, color=MID)
    for ax, (title, col) in zip(axes, [
        ("Primer semestre", "Unidades_curriculares_1er_semestre_(aprobadas)"),
        ("Segundo semestre", "Unidades_curriculares_2do_semestre_(aprobadas)"),
    ]):
        rates = [100 * (train.loc[train.target == label, col] == 0).mean() for label in groups]
        y = np.arange(len(groups))
        ax.barh(y, rates, height=0.52, color=palette, zorder=2)
        ax.set_yticks(y, [f"{g}  (n={sizes[g]:,})".replace(",", ".") for g in groups])
        ax.set_title(title, fontsize=12, loc="left", pad=13, color=DARK, weight="bold")
        style(ax, 78, [0, 25, 50, 75])
        for yi, value in zip(y, rates):
            ax.text(value + 1, yi, f"{value:.1f}%".replace(".", ","),
                    va="center", fontsize=10.5, color=DARK)
    axes[0].invert_yaxis()
    fig.text(0.05, 0.055, "Cero puede indicar falta de actividad registrada; no se conoce la fecha individual de abandono.",
             fontsize=9, color=MID)
    fig.subplots_adjust(left=0.255, right=0.98, top=0.71, bottom=0.21, wspace=0.16)
    fig.savefig(OUT / "06_eda_ceros_por_clase.svg", facecolor="white")
    plt.close(fig)

    # Pregunta: ¿qué asociaciones administrativas motivan revisar fechas?
    comparisons = [
        ("Matrícula no al día", "Pago_de_matrícula_al_día", "No"),
        ("Matrícula al día", "Pago_de_matrícula_al_día", "Sí"),
        ("Con deuda", "Tiene_deuda", "Sí"),
        ("Sin deuda", "Tiene_deuda", "No"),
        ("Sin beca", "Poseedor_de_Beca", "No"),
        ("Con beca", "Poseedor_de_Beca", "Sí"),
    ]
    rows = []
    for name, col, value in comparisons:
        sub = train.loc[train[col] == value]
        rows.append((name, len(sub), 100 * (sub.target == "Desertor").mean()))
    baseline = 100 * (train.target == "Desertor").mean()
    fig, ax = plt.subplots(figsize=(10.8, 4.65))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.94, "Las asociaciones administrativas son grandes", fontsize=16,
             weight="bold", color=DARK)
    fig.text(0.06, 0.875, "Porcentaje Desertor dentro de cada grupo · train, n = 40.000",
             fontsize=10.5, color=MID)
    y = np.arange(len(rows))
    ax.barh(y, [r[2] for r in rows], height=0.58,
            color=[RED if r[2] > baseline else MID for r in rows], zorder=2)
    ax.axvline(baseline, color=DARK, linewidth=1, linestyle=(0, (3, 3)), zorder=3)
    ax.set_yticks(y, [f"{name}  (n={n:,})".replace(",", ".") for name, n, _ in rows])
    ax.invert_yaxis()
    style(ax, 110, [0, 25, 50, 75, 100])
    for yi, (_, _, value) in zip(y, rows):
        ax.text(value + 1, yi, f"{value:.1f}%".replace(".", ","),
                va="center", fontsize=10.5, color=DARK, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    fig.text(0.06, 0.105, f"Línea: tasa de train completo, {baseline:.1f}%. Fecha de registro de estas variables: no acreditada.".replace("33.1", "33,1"),
             fontsize=9, color=MID)
    fig.subplots_adjust(left=0.32, right=0.97, top=0.74, bottom=0.19)
    fig.savefig(OUT / "07_eda_administrativas.svg", facecolor="white")
    plt.close(fig)

    print("Train:", len(train), "| Clases:", sizes.to_dict())
    print("Figuras:", OUT / "06_eda_ceros_por_clase.svg", OUT / "07_eda_administrativas.svg")


if __name__ == "__main__":
    main()
