# Trazabilidad del TP completo

Este archivo indica de dónde sale cada parte del borrador.

## Consigna

Fuente contrastada: [Formato informe 2026, Trabajo práctico 1](https://docs.google.com/document/d/1a7cHCdDvyGEcpK6Ek2F3Kir4JldNpJUJ/edit) (la guía indica que es orientativa, no la única forma correcta de hacer el informe).

Requisitos usados:
- máximo 12 carillas;
- resumen ≤200 palabras;
- introducción en hasta 3 párrafos;
- Materiales y métodos: datos, target y esquemas de clasificación;
- binario: Logística, SVM, Árbol + Voting;
- multiclase: KNN, Árbol; aclarar OvO/OvR como estrategia de clasificador sólo cuando corresponda, distinguiéndola del cálculo de ROC-AUC;
- al menos un Bagging y un Boosting;
- métricas, curvas de complejidad e importancia de atributos;
- discusión ≤1 carilla;
- conclusión de un párrafo.

## Datos / EDA

- Resolución TP AA/informe EDA/informe_eda.md
- Resolución TP AA/informe preprocesamiento/conclusiones_preprocesamiento.md

Números principales:
- n=50.000;
- 27 predictores;
- 11 numéricas / 16 categóricas;
- base completa: Graduado 23.707; Desertor 16.529; En Curso 9.764;
- train del EDA: n=40.000; Graduado 18.966, Desertor 13.223, En Curso 7.811;
- 0 faltantes en train;
- 213 filas idénticas en train (313 en la base completa);
- temporalidad: 12 candidatas de ingreso + 4 S1 + 4 S2 + 7 sin fecha acreditada;
- el EDA original y el anexo `AA privada/02_alternativa_temporal/eda/` usan sólo el mismo train estratificado por tres clases (semilla 42);
- una versión anterior del EDA original examinó también test: sus métricas de test actuales son exploratorias y no evidencia confirmatoria nueva.

## Binario

- Resolución TP AA/informe clasificación binaria/informe_clasificacion_binaria.md
- resultados/03_binario.csv

## Complejidad

- Resolución TP AA/informe complejidad y generalización/informe_complejidad.md
- resultados/04_complejidad/04_curvas_complejidad.csv

## Voting

- Resolución TP AA/informe Voting/informe_voting.md
- resultados/05_voting.csv

## Multiclase

- Resolución TP AA/informe multiclase/informe_multiclase.md
- resultados/06_multiclase.csv
- resultados/06_multiclase_por_clase.csv

## Bagging y Boosting

- Resolución TP AA/informe Bagging y Boosting/informe_ensembles.md
- resultados/07_ensembles.csv

## Importancia

- Resolución TP AA/informe importancia de atributos/informe_importancia.md
- resultados/08_importancia.csv
- resultados/08_importancia_bloques.csv

## Comparación final

- Resolución TP AA/informe comparación final/informe_comparacion_final.md
- resultados/09_comparacion_binaria.csv
- resultados/09_comparacion_multiclase.csv

## Bibliografía externa verificada

- Realinho et al., Data 2022, doi:10.3390/data7110146.
- Martins et al., Applied Sciences 2023, doi:10.3390/app13084702.
- DGES, sistema de enseñanza superior portugués.
- DGES, propinas.
- Proyecto PIAES del Politécnico de Portalegre.
- Kaggle Playground Series S4E6.
