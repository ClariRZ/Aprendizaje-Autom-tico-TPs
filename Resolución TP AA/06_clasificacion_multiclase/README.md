# 06 — Segunda tarea: conservar las tres clases

## Pregunta

¿Qué errores quedan ocultos cuando En Curso y Graduado se juntan bajo la etiqueta No desertor?

## Decisiones y explicación

El target ahora tiene **Desertor, En Curso y Graduado**. Se reserva otro 20 % de test con semilla 42 y estratificación por las tres clases; es el mismo protocolo del EDA, distinto del split binario. Un estudiante En Curso no es todavía graduado ni desertor: sin seguimiento no podemos inferir su desenlace.

`06_multiclase.py` entrena K-NN y Árbol. K-NN usa escala para que las distancias no dependan de las unidades originales; el Árbol usa umbrales y no la necesita. Ambos admiten tres clases directamente, sin envoltorio One-vs-Rest ni One-vs-One. La búsqueda usa tres folds de train y **F1 macro**, que calcula un F1 por clase y los promedia con igual peso; esa elección obliga a atender también a En Curso. K-NN prueba vecinos `3, 5, 9, 15, 25` y pesos uniforme/distancia; Árbol, las seis profundidades y tres mínimos de hoja del paso binario, pero **se vuelve a elegir** porque cambió el target. La selección final fue k=25 y peso uniforme, y Árbol con profundidad 12 y hoja 50.

La ROC-AUC macro OvR calculada desde las probabilidades es una **métrica de evaluación**, no una estrategia que transforme cómo se entrenó el modelo. Las métricas por clase y la matriz de confusión muestran si En Curso termina clasificado como Graduado o Desertor. Esto contrasta H3 del [informe final](../../entrega_final/TP1.md), secciones 1 y 3.5; la figura 6 y los valores de H3 corresponden al bosque del paso 07, no al Árbol de este script.

## Script y archivos

```bash
python "Resolución TP AA/06_clasificacion_multiclase/06_multiclase.py"
```

Genera en `Resolución TP AA/resultados/`: `06_multiclase.csv` (métricas finales y parámetros), `06_multiclase_cv.csv` (todos los ensayos), `06_multiclase_por_clase.csv` y `06_multiclase_confusion.csv` (conteos, no normalización). La tabla 4 del informe reúne estos resultados con los del paso 07. El test ya observado históricamente permite describir, no confirmar de forma independiente, H3.

