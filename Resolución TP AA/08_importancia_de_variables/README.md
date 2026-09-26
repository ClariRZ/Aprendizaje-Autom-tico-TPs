# 08 — ¿De qué columnas depende el bosque?

## Pregunta

Cuando el Random Forest multiclase predice, ¿qué variables **originales** hacen perder más F1 macro si se desordenan?

## Decisión y límites

`08_importancia.py` recrea el bosque seleccionado en 07 (500 árboles, sin profundidad máxima, hoja 1, `max_features="sqrt"`). Divide por las tres clases, aparta test y, dentro del 80 % de train, hace tres folds. En cada uno ajusta el pipeline en dos tercios y mide F1 macro en el tercio retenido. Luego `permutation_importance` mezcla **cada columna de X anterior al One-Hot** cinco veces y registra cuánto cae el F1. Así una variable categórica cambia como bloque; permutar dummies sueltas daría otra interpretación. No calcula importancia en test.

El programa agrupa las 27 columnas en académico, administrativo/económico e ingreso/contexto. Las ocho académicas informan H1; las administrativas y ciertas variables de ingreso, H2. **H3 se comprueba con la matriz de confusión y métricas por clase, no con un ranking de variables**. El script incorporado en esta copia actualiza únicamente las etiquetas de hipótesis del CSV para seguir la formulación del informe; cálculos, parámetros y métricas permanecen iguales.

Una columna muy asociada en el EDA puede aportar poco al modelo si otra contiene la misma información. Una caída pequeña por permutación tampoco demuestra irrelevancia cuando dos columnas son redundantes. Ni las caídas ni sus sumas por bloque establecen causalidad o el resultado de intervenir sobre una variable. En la sección 3.8 del [informe final](../../entrega_final/TP1.md), aprobadas de S2 presenta caída media 0,1004; matrícula al día, 0,0360. Las ocho académicas acumulan 0,1533 y las tres administrativas 0,0468; son sumas descriptivas, no partes aditivas del F1.

## Script y salida

```bash
python "Resolución TP AA/08_importancia_de_variables/08_importancia.py"
```

Escribe `08_importancia.csv` (ranking y etiquetas), `08_importancia_repeticiones.csv` (cinco repeticiones por fold), `08_importancia_bloques.csv` y `08_importancia_baseline_cv.csv` en `Resolución TP AA/resultados/`. Para reproducir la figura 8 se usan estas caídas medias. Ejecutarlo es más costoso que imprimir las tablas: requiere volver a entrenar tres bosques y evaluar 27 variables varias veces.

