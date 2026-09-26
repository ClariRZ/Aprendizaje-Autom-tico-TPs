# 07 — Dos maneras de combinar árboles

## Pregunta

¿Random Forest y AdaBoost mejoran la clasificación de las tres trayectorias respecto del Árbol individual del paso 06? ¿Cómo influyen sus mecanismos en el sobreajuste?

## Qué se eligió

Los dos usan el split multiclase del paso 06, con preprocesamiento dentro del pipeline, búsqueda por F1 macro en tres folds de train y evaluación final en el mismo test de 10.000 filas. Random Forest ajusta muchos árboles sobre muestras bootstrap y ofrece a cada división una selección aleatoria de columnas (`max_features="sqrt"`); promediarlos reduce la variación del Árbol individual, y elegir columnas al azar reduce la semejanza de los errores entre árboles. Gini considera simultáneamente las tres clases. AdaBoost agrega árboles cortos en secuencia y da más peso a observaciones difíciles. Ambos manejan tres clases de forma nativa; OvR en ROC-AUC se refiere sólo a la forma de medirla.

La grilla inicial del bosque prueba 200/500 árboles, profundidad 12/sin límite y hoja 1/50; la de AdaBoost prueba profundidad base 1/2, 100/200/400 árboles y tasa 0,05/0,1. Se seleccionaron bosque con 500 árboles, sin tope de profundidad y hoja 1, y AdaBoost con profundidad 2, 400 árboles, tasa 0,1. Estos son óptimos **dentro de esa grilla**, no óptimos universales: AdaBoost continuó mejorando al probar 600 árboles en una comprobación posterior de train sin sustituir el modelo final.

## Script y lectura

```bash
python "Resolución TP AA/07_bagging_y_boosting/07_ensembles.py"
```

En `Resolución TP AA/resultados/` escribe `07_ensembles.csv`, `07_ensembles_cv.csv`, `07_ensembles_por_clase.csv`, `07_ensembles_confusion.csv` y `07_comparacion_con_arbol.csv`. La última tabla incorpora el Árbol sólo si `06_multiclase.csv` ya existe. Los conteos de confusión se pueden normalizar por la cantidad real de cada clase para la figura 6 del [informe final](../../entrega_final/TP1.md). En su tabla 4, Random Forest obtiene F1 macro 0,7808 y AdaBoost 0,7776; el bosque es el modelo interpretado en el paso 08. Las curvas multiclase de la figura 4 y la ampliación hasta 600 árboles se obtuvieron en `AA privada/03_verificaciones_catedra/`, no en este script.

