# 02 — Preparar datos sin usar información del test

## Pregunta

¿Cómo convierto las 27 columnas originales en entradas numéricas para los clasificadores sin que la validación influya en el ajuste de transformaciones?

## Qué se decidió y por qué

`_comun.pre(X, escalar=True)` identifica numéricas y categóricas. Para numéricas usa mediana y, cuando el algoritmo depende de distancias o magnitudes (Logística, SVM, K-NN), estandarización. Para categóricas usa la moda y `OneHotEncoder(handle_unknown="ignore")`: una categoría nunca vista puede procesarse al evaluar nuevos datos. Para Árbol, Random Forest y AdaBoost se llama `pre(X, False)`, porque un árbol divide por umbrales y no necesita escalar.

Aunque esta base no tiene faltantes explícitos, la imputación permite reutilizar el procedimiento con entradas incompletas. Con la base actual, el One-Hot expande los 27 predictores a 77 columnas transformadas. El EDA no se usó para quitar columnas.

## Script y alcance

Desde la raíz, después de reconstruir la base:

```bash
python "Resolución TP AA/02_preprocesamiento_sin_fugas/02_preprocesamiento.py"
```

Este script es una **demostración**: divide 80/20 por el target binario, ajusta un transformador sólo sobre `Xtr`, transforma `Xte` y muestra tamaños y proporciones por pantalla. No produce un modelo ni un CSV final. Los pasos 03–08 implementan el preprocesamiento **dentro de cada `Pipeline`**: en cada fold de validación se recalculan medianas, modas, escalas y categorías usando sólo la parte de ajuste. Esa integración es la que respalda el método del [informe final](../../entrega_final/TP1.md), sección 2.3.

La separación train/test antecede al ajuste del modelo; durante la búsqueda de hiperparámetros el test queda reservado. Hay otra partición para tres clases, estratificada por esas etiquetas, de modo que los dos test no tienen exactamente las mismas filas.

