# Alternativa: información disponible por etapas

[Análisis de resultados frente a la primera resolución](../COMPARACION_RESULTADOS.md).

La guía principal busca **predecir deserción mediante clasificación binaria
y multiclase** con información de inscripción y del primer año; la
[primera resolución](../01_resolucion_original/README.md) decidió usar las
27 variables en el caso completo. La guía no obliga a usar todas. Este experimento por etapas es una extensión
exploratoria: compara cuánto añaden S1 y S2, sin convertir la consigna en
un requisito de predicción desde el ingreso.

## Pregunta

¿Qué capacidad predictiva muestra **la misma familia de modelo** cuando pasa
de variables candidatas al ingreso a resultados del primer y segundo semestre?
¿Cuánto cambia si añadimos columnas cuya fecha de medición no está documentada?

**Orden para aprender el análisis:** empezar por
[EDA en escalera](eda/RECORRIDO_EDA.md), el
[EDA comentado variable por variable](eda/EDA_27_VARIABLES_COMENTADO.md) y el
[atlas completo de las 27 variables](eda/atlas/README.md).
El [mapa de asociaciones](eda/MAPA_ASOCIACIONES_27.md) reúne en dos tablas
cuáles parecen relacionadas con el target, los números que lo justifican y
qué interpretación todavía falta comprobar.
Allí se pregunta primero qué datos tenemos y cómo se distribuyen, después
qué asociaciones aparecen con el target y qué columnas se solapan;
al final se presentan las dos vías: modelo con las 27 juntas o comparación
de los tres momentos de información. Las cinco figuras de
[síntesis](eda/README.md#cinco-gráficos-para-leer-las-hipótesis) sirven
para el informe breve.

## Etapas predefinidas

| Etapa | Columnas | Interpretación |
|---|---:|---|
| E0 | 12 | Candidatas a ingreso, excluyendo las siete sin fecha documentada. |
| E1 | 16 | E0 + cuatro variables con nombre de primer semestre. |
| E2 | 20 | E1 + cuatro variables con nombre de segundo semestre. |
| E0+?, E1+?, E2+? | 19, 23, 27 | Sensibilidad al agregar deuda, matrícula, beca y cuatro variables parentales sin fecha verificada. |

Los nombres de E0 son una clasificación provisional. Si se quiere presentar
un modelo **al ingreso** o una alerta anterior al abandono, hay que confirmar
qué campos estaban presentes entonces. La base no incluye la fecha individual
de deserción. Esta limitación temporal afecta esa aplicación adicional; el
modelo con el año completo sigue respondiendo la consigna del TP.

## Protocolo

1. Reconstruir la base con `python 00_base_de_datos/reconstruir_dataset.py`,
   si el CSV aún no existe. El CSV reconstruido no se vuelve a versionar aquí.
2. Separar una sola vez 40.000 train / 10.000 test, estratificado por las tres
   clases con semilla 42. El EDA condicionado por target usa solo train.
3. Formular hipótesis descriptivas en `eda/README.md` a partir de train.
   Son hipótesis **exploratorias**, porque
   este conjunto ya fue analizado para el TP original.
4. Mantener las mismas filas y folds entre etapas. Imputación, One-Hot y
   escalado (Logística) se aprenden dentro de los pipelines de cada fold.
5. GridSearch: binario Logística `C=[.01,.1,1,10]`, criterio ROC-AUC;
   multiclase Árbol `max_depth=[4,8,12,None]`,
   `min_samples_leaf=[10,50]`, criterio F1 macro. CV estratificada de 3 folds.
6. Reajustar la configuración elegida en todo train y evaluar el test una vez
   para cada combinación **preespecificada** de tarea y etapa. Guardar
   métricas, matrices de confusión y valores por fold. Las predicciones
   individuales se conservan localmente y no se versionan.
7. Como sensibilidad, repetir las métricas quitando del test los casos con
   las **27 columnas exactamente iguales** a alguna fila de train. Esto no
   garantiza independencia entre personas sin identificador.

## Ejecutar

```bash
python "AA privada/02_alternativa_temporal/eda/eda_train.py" --input 00_base_de_datos/dataset_TP1_2026.csv
python "AA privada/02_alternativa_temporal/eda/atlas_variables.py" --input 00_base_de_datos/dataset_TP1_2026.csv
python "AA privada/02_alternativa_temporal/modelos/comparar_etapas.py"
python "AA privada/02_alternativa_temporal/modelos/validacion_config_fija.py"
python "AA privada/02_alternativa_temporal/modelos/graficar_resultados.py"
```

Los resultados versionados fueron calculados sobre la base reconstruida por el
script del repositorio. El JSON de protocolo registra su hash SHA-256.

## Límites

La base es sintética y procede de una transformación de Kaggle/UCI. Las tasas
de deserción observadas describen esta base; no son riesgos causales para
estudiantes reales. E0, E1 y E2 pueden identificar distintos estados de una
trayectoria ya avanzada; las notas de segundo semestre no prueban que la
deserción pudiera anticiparse. La etiqueta «En Curso» tampoco es un desenlace
terminal. El test de esta base ya se vio en el TP original: resultados nuevos
sirven para explorar una pregunta alternativa y necesitan una nueva cohorte
para confirmar generalización externa.
