# Informe EDA

Esta carpeta contiene la versión narrativa y visual del EDA final.

**Conjunto analizado:** las mismas 40.000 filas de train del split multiclase 80/20 (semilla 42). Se reservan 10.000 para evaluación y no se usan para elegir variables o hipótesis. La primera versión histórica del EDA sí observó test; por ello los resultados en ese test no son evidencia confirmatoria nueva.

## Estado

**EDA cerrado.** Sólo debe reabrirse si aparece un error de datos o una inconsistencia que impida el modelado.

## Qué contiene

El informe resume:

1. procedencia y construcción de la base;
2. distribución de las tres clases;
3. disponibilidad temporal de los predictores;
4. inactividad académica y ceros;
5. rendimiento entre estudiantes activos;
6. señales tempranas como edad;
7. bloque administrativo/económico;
8. redundancia entre predictores;
9. limitaciones y posibles sesgos;
10. hipótesis que pasan al modelado.

## Gráficos

Los SVG se versionan en GitHub y se muestran directamente dentro de `informe_eda.md`.

## Regenerar

Primero ejecutar el EDA final y luego:

    python "Resolución TP AA/informe EDA/generar_graficos_informe.py"

## Regla interpretativa

Asociación descriptiva no implica causalidad ni equivale a importancia predictiva. Las hipótesis formuladas al final deben contrastarse con los modelos.
