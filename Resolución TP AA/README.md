# Resolución TP AA

Ruta principal del trabajo:

01 EDA exploratorio final → 02 preprocesamiento → 03 clasificación binaria → 04 curvas de complejidad → 05 Voting → 06 multiclase → 07 bagging/boosting → 08 importancia → 09 comparación → 10 informe

## Regla de cierre del EDA

El EDA original está en 01_entender_problema_y_datos. Sus tablas, figuras y asociaciones se
recalcularon sobre 40.000 filas de train, apartando antes las 10.000 de test
con semilla 42 y estratificación por las tres clases. La versión anterior examinó
toda la base; por eso el test ya observado no confirma hipótesis de manera
independiente. El EDA complementario usa el mismo train y está en
`AA privada/02_alternativa_temporal/eda/`.

Se considera cerrado cuando:
- se revisó calidad y estructura;
- se vieron distribuciones univariadas;
- se compararon las tres clases originales;
- se separó cero/inactividad de nivel positivo en variables académicas;
- se revisaron categorías pequeñas;
- se revisó redundancia entre predictores;
- se hizo el puente descriptivo hacia la tarea binaria;
- se discutió el archivo EDA_FINAL_PARA_EVALUAR.md.

Después de eso no se agregan análisis exploratorios salvo que aparezca un problema de datos.

## Reglas metodológicas

- asociación descriptiva no equivale a importancia predictiva ni causalidad;
- el EDA no selecciona variables por ranking;
- hacer el split antes de explorar la relación entre variables y target;
- preprocesamiento y selección de hiperparámetros se ajustan sólo dentro de train/folds;
- el test actual ya fue observado históricamente: no reutilizarlo para formular hipótesis ni ajustar modelos.
