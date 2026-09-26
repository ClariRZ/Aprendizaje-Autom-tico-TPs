# 09 — Comparar sin mezclar preguntas diferentes

## Pregunta

¿Qué modelo conviene describir como resultado de cada tarea y qué información se pierde al condensarlo en una métrica global?

## Decisión

`09_comparacion.py` **no entrena**: reúne CSV de los pasos 03, 05, 06 y 07. Compara Logística, SVM, Árbol y Voting en binario, y K-NN, Árbol, Random Forest y AdaBoost en multiclase. Ordena el primero por ROC-AUC y el segundo por F1 macro, las métricas de selección respectivas. Sólo dentro de cada tarea los modelos tienen el mismo test y predicen las mismas etiquetas. Comparar directamente un F1 binario con uno multiclase, o AUC binaria con AUC OvR de tres clases, respondería preguntas distintas.

Se agrega una columna con la diferencia entre CV y test, útil para inspeccionar estabilidad, **sin** usarla para elegir de nuevo hiperparámetros. Voting no hizo una búsqueda equivalente a la de los modelos individuales y por eso carece de valor CV comparable en la tabla. La decisión del [informe final](../../entrega_final/TP1.md), secciones 3.2 y 3.5, mira además sensibilidad por clase, confusión y curvas: Logística logra el mejor F1 binario (0,8510); Voting, la mayor AUC binaria (0,9514), aunque con ganancia mínima; Random Forest alcanza el mejor F1 macro de las tres clases (0,7808).

## Script y requisitos

```bash
python "Resolución TP AA/09_comparacion_final/09_comparacion.py"
```

Antes tienen que existir `03_binario.csv`, `05_voting.csv`, `06_multiclase.csv` y `07_ensembles.csv` en `Resolución TP AA/resultados/`; el script detiene la ejecución si falta uno. Genera `09_comparacion_binaria.csv`, `09_comparacion_multiclase.csv` y `09_resumen_comparacion.md`. Las tablas 2 y 4 del informe son la presentación redactada de las métricas. La diferencia CV-test mezcla conjuntos con tamaños distintos y no prueba por sí sola sobreajuste; para eso se estudian también las curvas del paso 04.

