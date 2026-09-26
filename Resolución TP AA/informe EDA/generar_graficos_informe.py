from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R))

from _comun import cargar_eda_train, OUT

TAB = OUT / "eda_exploratorio" / "tablas"
FIG = HERE / "graficos"
FIG.mkdir(parents=True, exist_ok=True)

required = [
    "02_target_tres_clases.csv",
    "11_asociacion_categoricas_target3.csv",
    "12_academicas_ceros_y_positivos.csv",
    "14_puente_binario_categorias.csv",
    "16_pares_numericos.csv",
]
missing = [x for x in required if not (TAB / x).exists()]
if missing:
    raise FileNotFoundError(
        "Primero ejecutá el EDA final. Faltan: " + ", ".join(missing)
    )

CLASES = ["Graduado", "En Curso", "Desertor"]

# ------------------------------------------------------------------
# 1. Target
# ------------------------------------------------------------------
target = pd.read_csv(TAB / "02_target_tres_clases.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(target["clase"], target["porcentaje"])
ax.set_ylabel("% de la muestra")
ax.set_title("La muestra contiene tres desenlaces académicos")
ax.set_ylim(0, max(target["porcentaje"]) * 1.18)
for i, v in enumerate(target["porcentaje"]):
    ax.text(i, v + 0.8, f"{v:.1f}%", ha="center")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "01_target_tres_clases.svg", dpi=180)
plt.close(fig)

# ------------------------------------------------------------------
# 2. Ceros en variables académicas clave
# ------------------------------------------------------------------
acad = pd.read_csv(TAB / "12_academicas_ceros_y_positivos.csv")
key = [
    "Unidades_curriculares_1er_semestre_(aprobadas)",
    "Nota_promedio_en_el_1er_semestre",
    "Unidades_curriculares_2do_semestre_(aprobadas)",
    "Nota_promedio_en_el_2do_semestre",
]
labels = ["Aprobadas S1", "Nota S1", "Aprobadas S2", "Nota S2"]

x = np.arange(len(key))
w = 0.24
fig, ax = plt.subplots(figsize=(9, 5.2))
for k, clase in enumerate(CLASES):
    s = acad[acad["clase"] == clase].set_index("variable").loc[key]
    ax.bar(x + (k - 1) * w, s["pct_ceros"], w, label=clase)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("% con valor 0")
ax.set_title("La inactividad académica separa con fuerza a los desertores")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "02_ceros_academicos.svg", dpi=180)
plt.close(fig)

# ------------------------------------------------------------------
# 3. Nivel entre estudiantes con valores positivos
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
groups = [
    (["Unidades_curriculares_1er_semestre_(aprobadas)",
      "Unidades_curriculares_2do_semestre_(aprobadas)"],
     ["Aprobadas S1", "Aprobadas S2"], "Materias aprobadas"),
    (["Nota_promedio_en_el_1er_semestre",
      "Nota_promedio_en_el_2do_semestre"],
     ["Nota S1", "Nota S2"], "Nota promedio"),
]

for ax, (vars_, labs, title) in zip(axes, groups):
    x = np.arange(len(vars_))
    for k, clase in enumerate(CLASES):
        s = acad[acad["clase"] == clase].set_index("variable").loc[vars_]
        ax.bar(x + (k - 1) * w, s["mediana_positivos"], w, label=clase)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Mediana entre valores > 0")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
fig.suptitle("Entre quienes tienen actividad, también difiere el nivel", y=.98)
fig.tight_layout(rect=[0, 0, 1, .90])
fig.savefig(FIG / "03_nivel_entre_activos.svg", dpi=180)
plt.close(fig)

# ------------------------------------------------------------------
# 4. Edad en la tarea binaria
# ------------------------------------------------------------------
df = cargar_eda_train()
age = "Edad_al_momento_de_la_inscripción"
es_desertor = df["target"].eq("Desertor")
medianas = [
    df.loc[~es_desertor, age].median(),
    df.loc[es_desertor, age].median(),
]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(["No desertor", "Desertor"], medianas)
ax.set_ylabel("Mediana de edad")
ax.set_title("La edad aporta una señal moderada desde el ingreso")
for i, v in enumerate(medianas):
    ax.text(i, v + 0.4, f"{v:.0f} años", ha="center")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "04_edad_por_clase.svg")
plt.close(fig)

# ------------------------------------------------------------------
# 5. Asociación categórica global
# ------------------------------------------------------------------
cv = pd.read_csv(TAB / "11_asociacion_categoricas_target3.csv").head(8)
cv = cv.sort_values("cramers_v_target3")
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(cv["variable"], cv["cramers_v_target3"])
ax.set_xlabel("Cramér's V con Graduado / En Curso / Desertor")
ax.set_title("Las asociaciones categóricas más fuertes son administrativas y contextuales")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "05_asociacion_categoricas.svg", dpi=180, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# 6. Tres contrastes administrativos vs tasa global
# ------------------------------------------------------------------
bc = pd.read_csv(TAB / "14_puente_binario_categorias.csv")
wanted = [
    ("Pago_de_matrícula_al_día", "No", "Matrícula no al día"),
    ("Tiene_deuda", "Sí", "Tiene deuda"),
    ("Poseedor_de_Beca", "Sí", "Beca = Sí"),
]
rows = []
for variable, categoria, label in wanted:
    s = bc[(bc["variable"] == variable) & (bc["categoria"] == categoria)]
    if len(s):
        rows.append((label, 100 * float(s.iloc[0]["tasa_desercion"])))

global_rate = 100 * (df["target"] == "Desertor").mean()
admin = pd.DataFrame(rows, columns=["condicion", "tasa"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(admin["condicion"], admin["tasa"])
ax.axhline(global_rate, linestyle="--", linewidth=1.5,
           label=f"Tasa global {global_rate:.1f}%")
ax.set_ylabel("Tasa de deserción (%)")
ax.set_ylim(0, 100)
ax.set_title("Las variables administrativas muestran contrastes grandes")
for i, v in enumerate(admin["tasa"]):
    ax.text(i, v + 2, f"{v:.1f}%", ha="center")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "06_administrativas_desercion.svg", dpi=180)
plt.close(fig)

# ------------------------------------------------------------------
# 7. Redundancia académica
# ------------------------------------------------------------------
pairs = pd.read_csv(TAB / "16_pares_numericos.csv")
mask = (
    pairs["variable_1"].str.contains("semestre|Nota_promedio", regex=True) &
    pairs["variable_2"].str.contains("semestre|Nota_promedio", regex=True)
)
p = pairs[mask].head(7).copy()
def short(s):
    return (s.replace("Unidades_curriculares_1er_semestre_(inscrito)", "Inscriptas S1")
             .replace("Unidades_curriculares_2do_semestre_(inscrito)", "Inscriptas S2")
             .replace("Unidades_curriculares_1er_semestre_(aprobadas)", "Aprobadas S1")
             .replace("Unidades_curriculares_2do_semestre_(aprobadas)", "Aprobadas S2")
             .replace("Nota_promedio_en_el_1er_semestre", "Nota S1")
             .replace("Nota_promedio_en_el_2do_semestre", "Nota S2")
             .replace("Cantidad_de_evaluaciones_en_el_1er_semestre", "Evaluaciones S1")
             .replace("Cantidad_de_evaluaciones_en_el_2do_semestre_", "Evaluaciones S2"))
p["par"] = p["variable_1"].map(short) + " ↔ " + p["variable_2"].map(short)
p = p.sort_values("abs_rho")

fig, ax = plt.subplots(figsize=(9, 5.3))
ax.barh(p["par"], p["abs_rho"])
ax.set_xlabel("|ρ de Spearman|")
ax.set_xlim(0, 1)
ax.set_title("Varias variables académicas contienen información redundante")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "07_redundancia_academica.svg", dpi=180, bbox_inches="tight")
plt.close(fig)

print("Informe gráfico generado en:")
print(FIG)


# ------------------------------------------------------------------
# 8. Disponibilidad temporal de la información
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.axis("off")

xs = [0.08, 0.38, 0.66, 0.90]
titles = ["Ingreso candidato", "1.er semestre", "2.º semestre", "Desenlace"]
subs = [
    "12 candidatas\nfecha real por verificar",
    "+ 4 predictores\ninscriptas · evaluaciones\naprobadas · nota",
    "+ 4 predictores\ninscriptas · evaluaciones\naprobadas · nota",
    "Target\nGraduado · En Curso\nDesertor",
]

ax.annotate("", xy=(0.94, 0.52), xytext=(0.06, 0.52),
            arrowprops=dict(arrowstyle="->", linewidth=2))
for x, title, sub in zip(xs, titles, subs):
    ax.scatter([x], [0.52], s=700)
    ax.text(x, 0.72, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(x, 0.30, sub, ha="center", va="center", fontsize=10)

ax.text(0.5, 0.07,
        "Además hay 7 columnas sin fecha acreditada: deuda, matrícula, beca y cuatro parentales.",
        ha="center", va="center", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "08_disponibilidad_temporal.svg", bbox_inches="tight")
plt.close(fig)
