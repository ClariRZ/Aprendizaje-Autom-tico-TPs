# Concordancia de la versión abreviada

Se revisó la concordancia del texto con la base, los CSV de resultados,
las figuras y la reproducción independiente ya registrada. La revisión
de redacción no cambió modelos, particiones ni métricas.

## Acuerdos comprobados

- Las 40 métricas de las tablas principales coinciden, a cuatro
  decimales, con `metricas_reproducidas.csv`.
- Los rangos y valores seleccionados de la tabla de hiperparámetros se
  conservaron.
- Los conteos de clases y las tasas exploratorias reportadas concuerdan
  con el entrenamiento de la base de 50.000 filas.
- Los ocho puntajes del análisis acumulativo concuerdan con
  `AA privada/02_alternativa_temporal/resultados_modelos/metricas.csv` y
  las etiquetas de la Figura 7, considerando sus distintos redondeos.
- Las cifras de ampliación de grillas y sensibilidad de folds concuerdan
  con sus CSV. Los valores de las tablas con y sin repetidos también.
- La matriz de RF y sus errores de En Curso coinciden con la matriz
  publicada y la reproducción independiente del bosque.
- Las dos principales importancias reportadas concuerdan con la Figura 8.
  Se conserva su carácter descriptivo, sin interpretarlas como causas.

## Aclaraciones incorporadas

La comparación principal utiliza los 27 predictores y ambos semestres.
El análisis adicional agrega grupos (12 → 16 → 20 → 27), no compara
semestres aislados. Conserva la familia de modelo, pero selecciona
hiperparámetros en cada etapa; no aísla el efecto de agregar variables a
una configuración fija. El grupo administrativo y familiar no representa
una fecha posterior al segundo semestre.

Los modelos finales, las curvas con submuestras y los modelos del análisis
por bloques se identifican por separado. También se explica por qué la
AUC de Logística de la tabla principal puede diferir de la del análisis
por bloques: utilizan splits distintos.

## Alcance

Esta revisión reutiliza la reproducción anterior de los modelos finales.
No repitió todas las grillas ni las 405 permutaciones de importancia.
Para estas últimas se comprobó el acuerdo del texto con la figura y el
procedimiento publicado, no se realizó una nueva estimación completa.
El test permanece exploratorio por su observación previa. El número de
carillas depende del formato de exportación; el límite de 12 debe
comprobarse en la versión maquetada, y el campo de integrantes sigue
pendiente de completar.
