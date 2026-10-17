# Evidencias de verificación y corrección

La [revisión de concordancia de la versión abreviada](concordancia_redaccion.md)
detalla el acuerdo entre texto, tablas, figuras y resultados disponibles.

Se reconstruyó la base desde las diez partes CSV del repositorio y se
comparó su SHA-256 con el registrado en los protocolos existentes. Con
scikit-learn 1.8.0 se reentrenaron las configuraciones finales de los ocho
modelos: las 40 métricas de las tablas 2 y 4 se reprodujeron al redondear a
cuatro decimales. También se repitieron las grillas de Logística, Árbol
binario y Árbol multiclase, y la comparación de tres y cinco folds con
modelos fijos. No se repitieron todas las grillas de los demás modelos ni
la importancia por permutación completa.

## Corrección de la calibración

En `05_voting.py`, `CalibratedClassifierCV` ahora recibe el pipeline de
preprocesamiento y SVM completo. Antes, el preprocesamiento se ajustaba
antes de sus particiones internas. La corrección evita que el ajuste de
las transformaciones vea el fold reservado para calibración. Se mantuvo
el split, la semilla, los hiperparámetros y los pesos iguales; no se
eligió ninguna configuración mirando el test.

Se ejecutó nuevamente el script corregido. Accuracy, precisión, recall,
F1, AUC y conteos de confusión del Voting no cambiaron respecto de la
reproducción anterior. La AUC fue 0,9513504387004208 y el F1,
0,8471841213490399. La corrección no vuelve independiente al test ya
observado: la evaluación sigue siendo exploratoria.

## Archivos

- `metricas_reproducidas.csv`: métricas de los ocho modelos finales; el
  Voting corresponde a la ejecución del script corregido.
- `voting_calibracion_corregida.csv`: salida de esa ejecución, con matriz
  de confusión y duración.
- `sensibilidad_folds.csv`: medias y desvíos entre folds recalculados
  para modelos fijos. El desvío usa `ddof=0`; no es un intervalo de
  confianza y los folds no son independientes. El bosque reducido usa
  12.000 filas, 80 árboles, profundidad 12 y hoja mínima 10.
- `protocolo.json`: huella de la base, versiones, alcance y elecciones
  reproducidas de las tres grillas.

Los scripts originales y los diez CSV permiten repetir los procedimientos.
Las cifras de resultados describen esta base sintética y las particiones
publicadas; no validan una alerta temprana ni su transferencia a otra
cohorte.
