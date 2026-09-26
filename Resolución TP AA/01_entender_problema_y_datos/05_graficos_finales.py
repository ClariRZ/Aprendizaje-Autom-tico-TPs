from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from _comun import cargar_eda_train, OUT, SEMILLA
from eda_utils import CLASES, ACADEMICAS, ACADEMICAS_CLAVE, ADMIN_CLAVE, etiqueta

BASE = OUT / "eda_exploratorio"
TAB = BASE / "tablas"
FIG = BASE / "graficos"
FIG.mkdir(parents=True, exist_ok=True)

df = cargar_eda_train()

# 1. Target
target = pd.read_csv(TAB / "02_target_tres_clases.csv")
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(target["clase"], target["porcentaje"])
ax.set_ylabel("Porcentaje")
ax.set_title("Distribución del desenlace académico")
for i,v in enumerate(target["porcentaje"]):
    ax.text(i, v + .6, f"{v:.1f}%", ha="center")
fig.tight_layout()
fig.savefig(FIG / "01_target_tres_clases.png", dpi=180)
plt.close(fig)

# 2. Atlas numérico
num_cols = df.drop(columns="target").select_dtypes(include=np.number).columns.tolist()
ncols = 3
nrows = math.ceil(len(num_cols)/ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5*nrows))
axes = np.array(axes).reshape(-1)
for ax,c in zip(axes, num_cols):
    ax.hist(df[c].dropna(), bins=30)
    ax.set_title(etiqueta(c))
    ax.set_ylabel("Frecuencia")
for ax in axes[len(num_cols):]:
    ax.axis("off")
fig.suptitle("Atlas de distribuciones numéricas", y=1.01)
fig.tight_layout()
fig.savefig(FIG / "02_atlas_numericas.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 3. Numéricas con mayor separación entre tres clases
rank = pd.read_csv(TAB / "09_resumen_numericas_tres_clases.csv").head(6)
fig, axes = plt.subplots(2,3,figsize=(16,9))
for ax,c in zip(axes.ravel(), rank["variable"]):
    datos = [df.loc[df["target"]==cl,c].dropna() for cl in CLASES]
    ax.boxplot(datos, tick_labels=CLASES, showfliers=False)
    ax.set_title(etiqueta(c))
    ax.tick_params(axis="x", rotation=15)
fig.suptitle("Variables numéricas que más difieren entre las tres clases")
fig.tight_layout()
fig.savefig(FIG / "03_numericas_por_clase.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 4. Ceros académicos por clase
acad = pd.read_csv(TAB / "12_academicas_ceros_y_positivos.csv")
x = np.arange(len(ACADEMICAS))
width = .24
fig, ax = plt.subplots(figsize=(15,6))
for k,cl in enumerate(CLASES):
    s = acad[acad["clase"]==cl].set_index("variable").loc[ACADEMICAS]
    ax.bar(x + (k-1)*width, s["pct_ceros"], width, label=cl)
ax.set_xticks(x)
ax.set_xticklabels([etiqueta(c) for c in ACADEMICAS], rotation=35, ha="right")
ax.set_ylabel("% con valor 0")
ax.set_title("¿El contraste académico proviene de inactividad/ceros?")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "04_ceros_academicos_por_clase.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 5. Nivel entre positivos para cuatro variables académicas clave
fig, axes = plt.subplots(2,2,figsize=(13,9))
for ax,c in zip(axes.ravel(), ACADEMICAS_CLAVE):
    datos = []
    for cl in CLASES:
        xcl = df.loc[df["target"]==cl,c]
        datos.append(xcl[xcl>0])
    ax.boxplot(datos, tick_labels=CLASES, showfliers=False)
    ax.set_title(etiqueta(c) + " | sólo valores > 0")
    ax.tick_params(axis="x", rotation=15)
fig.suptitle("Cuando hay actividad, ¿también difiere el nivel?")
fig.tight_layout()
fig.savefig(FIG / "05_academicas_entre_positivos.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 6. Administrativas binarias: composición de las tres clases
fig, axes = plt.subplots(2,2,figsize=(13,9))
for ax,c in zip(axes.ravel(), ADMIN_CLAVE):
    tab = pd.crosstab(df[c], df["target"], normalize="index") * 100
    tab = tab.reindex(columns=CLASES, fill_value=0)
    bottom = np.zeros(len(tab))
    xpos = np.arange(len(tab))
    for cl in CLASES:
        vals = tab[cl].to_numpy()
        ax.bar(xpos, vals, bottom=bottom, label=cl)
        bottom += vals
    ax.set_xticks(xpos)
    ax.set_xticklabels(tab.index.astype(str), rotation=15)
    ax.set_ylim(0,100)
    ax.set_title(etiqueta(c))
    ax.set_ylabel("% dentro de categoría")
handles, labels = axes.ravel()[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)
fig.suptitle("Composición del desenlace en variables administrativas", y=.98)
fig.tight_layout(rect=[0,0,1,.94])
fig.savefig(FIG / "06_administrativas_tres_clases.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 7. Modo de aplicación y carrera
fig, axes = plt.subplots(1,2,figsize=(18,8))
for ax,c in zip(axes, ["modo_aplicacion","Carrera"]):
    counts = df[c].value_counts()
    cats = counts[counts >= 100].index
    tab = pd.crosstab(df[c],df["target"],normalize="index").loc[cats] * 100
    tab = tab.reindex(columns=CLASES, fill_value=0)
    ypos = np.arange(len(tab))
    left = np.zeros(len(tab))
    for cl in CLASES:
        vals=tab[cl].to_numpy()
        ax.barh(ypos, vals, left=left, label=cl)
        left += vals
    ax.set_yticks(ypos)
    ax.set_yticklabels(tab.index.astype(str), fontsize=8)
    ax.set_xlim(0,100)
    ax.set_xlabel("% dentro de categoría")
    ax.set_title(etiqueta(c))
fig.suptitle("Desenlace según modo de aplicación y carrera")
fig.tight_layout()
fig.savefig(FIG / "07_modo_y_carrera.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 8. Correlaciones numéricas
corr = pd.read_csv(TAB / "15_correlacion_spearman.csv", index_col=0)
fig, ax = plt.subplots(figsize=(11,9))
im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels([etiqueta(c) for c in corr.columns], rotation=90, fontsize=8)
ax.set_yticklabels([etiqueta(c) for c in corr.index], fontsize=8)
ax.set_title("Correlación de Spearman entre predictores numéricos")
fig.colorbar(im, ax=ax, shrink=.8)
fig.tight_layout()
fig.savefig(FIG / "08_correlacion_numericas.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 9. Redundancia categórica
cp = pd.read_csv(TAB / "18_pares_categoricos.csv").head(12).copy()
cp["par"] = cp["variable_1"].map(etiqueta) + " ↔ " + cp["variable_2"].map(etiqueta)
cp = cp.sort_values("cramers_v")
fig, ax = plt.subplots(figsize=(11,7))
ax.barh(cp["par"], cp["cramers_v"])
ax.set_xlabel("Cramér's V")
ax.set_title("Pares de predictores categóricos más asociados")
fig.tight_layout()
fig.savefig(FIG / "09_redundancia_categorica.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 10. Aprobadas vs nota, muestra reproducible para legibilidad
sample = df.sample(n=min(12000,len(df)), random_state=SEMILLA)
fig, axes = plt.subplots(1,2,figsize=(14,6))
for ax,sem in zip(axes,[1,2]):
    xc = f"Unidades_curriculares_{sem}{'er' if sem==1 else 'do'}_semestre_(aprobadas)"
    yc = f"Nota_promedio_en_el_{sem}{'er' if sem==1 else 'do'}_semestre"
    for cl in CLASES:
        s = sample[sample["target"]==cl]
        ax.scatter(s[xc],s[yc],s=8,alpha=.25,label=cl)
    ax.set_xlabel(etiqueta(xc))
    ax.set_ylabel(etiqueta(yc))
    ax.set_title(f"Semestre {sem}")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="upper center",ncol=3)
fig.suptitle("Relación entre materias aprobadas y nota")
fig.tight_layout(rect=[0,0,1,.94])
fig.savefig(FIG / "10_aprobadas_vs_nota.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"OK -> {FIG}")
