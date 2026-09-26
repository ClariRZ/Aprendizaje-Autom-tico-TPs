from pathlib import Path
import sys
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from _comun import OUT
from eda_utils import etiqueta

BASE = OUT / "eda_exploratorio"
TAB = BASE / "tablas"

base = pd.read_csv(TAB/"01_resumen_base.csv").iloc[0]
target = pd.read_csv(TAB/"02_target_tres_clases.csv")
checks = pd.read_csv(TAB/"04_chequeos_coherencia.csv")
uni_num = pd.read_csv(TAB/"05_univariado_numericas.csv")
uni_cat = pd.read_csv(TAB/"06_resumen_categoricas.csv")
num3 = pd.read_csv(TAB/"08_numericas_por_tres_clases.csv")
rank_num = pd.read_csv(TAB/"09_resumen_numericas_tres_clases.csv")
cat3 = pd.read_csv(TAB/"10_categorias_por_tres_clases.csv")
assoc_cat = pd.read_csv(TAB/"11_asociacion_categoricas_target3.csv")
acad = pd.read_csv(TAB/"12_academicas_ceros_y_positivos.csv")
bin_num = pd.read_csv(TAB/"13_puente_binario_numericas.csv")
bin_cat = pd.read_csv(TAB/"14_puente_binario_categorias.csv")
pairs_num = pd.read_csv(TAB/"16_pares_numericos.csv")
pairs_cat = pd.read_csv(TAB/"18_pares_categoricos.csv")

# formatos legibles para el informe
for d in [target]:
    if "porcentaje" in d:
        d["porcentaje"] = d["porcentaje"].round(2)

top_num = rank_num.head(8).copy()
top_num["eta2_tres_clases"] = top_num["eta2_tres_clases"].round(3)
top_num["rango_mediana"] = top_num["rango_mediana"].round(2)
top_num["max_gap_ceros_pp"] = top_num["max_gap_ceros_pp"].round(2)

top_cat = assoc_cat.head(10).copy()
top_cat["cramers_v_target3"] = top_cat["cramers_v_target3"].round(3)

ceros = acad.pivot(index="variable",columns="clase",values="pct_ceros").reset_index()
positivos = acad.pivot(index="variable",columns="clase",values="mediana_positivos").reset_index()
for c in ceros.columns[1:]:
    ceros[c] = ceros[c].round(2)
for c in positivos.columns[1:]:
    positivos[c] = positivos[c].round(2)

rare = uni_cat[uni_cat["n_categorias_menor_100"]>0].copy()
zero_flags = uni_num[uni_num["flag_muchos_ceros"]==True][["variable","pct_ceros","asimetria"]].copy()
zero_flags["pct_ceros"]=zero_flags["pct_ceros"].round(2)
zero_flags["asimetria"]=zero_flags["asimetria"].round(2)

pn = pairs_num.head(12).copy()
pn["rho_spearman"]=pn["rho_spearman"].round(3)
pc = pairs_cat.head(12).copy()
pc["cramers_v"]=pc["cramers_v"].round(3)

bn = bin_num.head(8).copy()
bn["smd"]=bn["smd"].round(3)
bn["pct_ceros_no_desertor"]=bn["pct_ceros_no_desertor"].round(2)
bn["pct_ceros_desertor"]=bn["pct_ceros_desertor"].round(2)

bc = bin_cat[bin_cat["n"]>=100].head(12).copy()
bc["tasa_desercion"]=(100*bc["tasa_desercion"]).round(2)
bc["diferencia_pp_vs_global"]=bc["diferencia_pp_vs_global"].round(2)

parts = [
"# EDA FINAL PARA EVALUAR","",
"Este EDA se calculó únicamente sobre las 40.000 filas de train, tras reservar 10.000 de test mediante split estratificado por tres clases (semilla 42). El test no interviene en distribuciones ni asociaciones. Resume lo observado antes de formular hipótesis y modelar.","",
"## 1. Base y calidad","",
f"- Filas: {int(base['filas']):,}.",
f"- Predictores: {int(base['predictores'])}.",
f"- Numéricas: {int(base['numericas'])}; categóricas: {int(base['categoricas'])}.",
f"- Faltantes: {int(base['faltantes_totales'])}.",
f"- Filas idénticas: {int(base['filas_identicas'])}.","",
"Chequeos de coherencia que deben quedar documentados, no corregidos automáticamente:","",
checks.to_markdown(index=False),"",
"## 2. Variable respuesta original","",
target.to_markdown(index=False),"",
"Primero exploramos Graduado, En Curso y Desertor por separado. El colapso a Desertor vs No desertor se usa sólo como puente hacia la primera tarea de clasificación.","",
"![Target](graficos/01_target_tres_clases.png)","",
"## 3. Forma de las variables","",
"Las tablas 05–07 contienen todas las distribuciones. Estas son las numéricas con al menos 20% de ceros:","",
zero_flags.to_markdown(index=False) if len(zero_flags) else "Ninguna numérica supera 20% de ceros.","",
"Variables categóricas con al menos una categoría de menos de 100 casos:","",
rare.to_markdown(index=False) if len(rare) else "No hay categorías con menos de 100 casos.","",
"Ver también: graficos/02_atlas_numericas.png","",
"## 4. Diferencias entre Graduado / En Curso / Desertor","",
"Para numéricas usamos eta² sólo como medida descriptiva de separación entre las tres clases; no es una prueba de causalidad ni un criterio de selección.","",
top_num.to_markdown(index=False),"",
"![Numéricas por clase](graficos/03_numericas_por_clase.png)","",
"Para categóricas usamos Cramér's V como asociación global con el target de tres clases:","",
top_cat.to_markdown(index=False),"",
"Ver graficos/06_administrativas_tres_clases.png y graficos/07_modo_y_carrera.png.","",
"## 5. La cuestión de los ceros en variables académicas","",
"Antes de interpretar medianas de 0 como bajo rendimiento, separamos dos fenómenos: presencia de valor 0 y nivel entre quienes tienen valores positivos.","",
"### Porcentaje de ceros por clase","",
ceros.to_markdown(index=False),"",
"### Mediana entre valores positivos","",
positivos.to_markdown(index=False),"",
"Ver graficos/04_ceros_academicos_por_clase.png y graficos/05_academicas_entre_positivos.png.","",
"## 6. Redundancia entre predictores","",
"### Pares numéricos","",
pn.to_markdown(index=False),"",
"### Pares categóricos","",
pc.to_markdown(index=False),"",
"Esto permite reconocer bloques de información repetida sin eliminar variables todavía.","",
"Ver graficos/08_correlacion_numericas.png y graficos/09_redundancia_categorica.png.","",
"## 7. Puente hacia la clasificación binaria del TP","",
"Recién aquí colapsamos Graduado + En Curso como No desertor para describir la tarea binaria.","",
"### Numéricas","",
bn[["variable","mediana_no_desertor","mediana_desertor","pct_ceros_no_desertor","pct_ceros_desertor","smd"]].to_markdown(index=False),"",
"### Categorías con n >= 100 y mayores diferencias respecto de la tasa global","",
bc[["variable","categoria","n","tasa_desercion","diferencia_pp_vs_global"]].to_markdown(index=False),"",
"## 8. Gráficos que debemos mirar antes de formular hipótesis","",
"1. 01_target_tres_clases.png — balance y tamaño de las tres clases.",
"2. 02_atlas_numericas.png — forma, ceros, asimetrías y rangos.",
"3. 03_numericas_por_clase.png — diferencias entre las tres clases.",
"4. 04_ceros_academicos_por_clase.png — cuánto del contraste académico es inactividad.",
"5. 05_academicas_entre_positivos.png — si el nivel también cambia cuando hay actividad.",
"6. 06_administrativas_tres_clases.png — matrícula, deuda, beca y sexo.",
"7. 07_modo_y_carrera.png — heterogeneidad entre categorías multivaluadas.",
"8. 08_correlacion_numericas.png — redundancia numérica.",
"9. 09_redundancia_categorica.png — redundancia categórica.",
"10. 10_aprobadas_vs_nota.png — estructura conjunta del bloque académico.","",
"## 9. Preguntas finales para nuestra discusión","",
"- ¿En Curso se parece más a Graduado o a Desertor, y en qué variables?",
"- ¿Las grandes diferencias académicas se deben principalmente a ceros/inactividad, a menor nivel entre activos, o a ambos?",
"- ¿Matrícula, deuda y beca muestran patrones distintos o podrían pertenecer a un mismo bloque administrativo/económico?",
"- ¿Qué señales ya están disponibles al ingreso y cuáles sólo aparecen durante el primer año?",
"- ¿Qué variables parecen redundantes entre sí?",
"- ¿Hay categorías con diferencias extremas pero tamaños pequeños que no deberían protagonizar la interpretación?",
"- ¿Cuáles 3–5 observaciones tienen suficiente coherencia para transformarse en hipótesis del trabajo?","",
"## 10. Criterio de cierre","",
"Si podemos responder las preguntas anteriores con estas tablas y gráficos, el EDA queda cerrado. Lo siguiente es formular hipótesis y pasar a preprocesamiento/modelado; no ampliar el EDA salvo que encontremos un error de datos."
]

out=BASE/"EDA_FINAL_PARA_EVALUAR.md"
out.write_text("\n".join(parts), encoding="utf-8")
print(f"OK -> {out}")
