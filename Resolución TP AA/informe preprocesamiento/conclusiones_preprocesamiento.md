# Conclusiones — Preprocesamiento sin fugas

## Objetivo

Preparar las variables para los modelos de clasificación sin permitir que información del conjunto de prueba influya en las transformaciones aprendidas durante el entrenamiento.

## Partición de los datos

La base completa contiene 50.000 estudiantes y 27 predictores.

Se realizó una partición estratificada 80/20:

| Conjunto | Filas | Predictores |
|---|---:|---:|
| Train | 40.000 | 27 |
| Test | 10.000 | 27 |

La estratificación preservó prácticamente de forma exacta la prevalencia de la clase Desertor:

| Conjunto | Proporción Desertor |
|---|---:|
| Base total | 33,058% |
| Train | 33,0575% |
| Test | 33,06% |

Esto permite comparar train y test sin introducir una diferencia artificial en la composición de clases.

## Transformación de variables

Los 27 predictores originales contienen variables numéricas y categóricas.

El pipeline aplica:

- **numéricas**: imputación por mediana y estandarización cuando el algoritmo es sensible a escala;
- **categóricas**: imputación por moda y One-Hot Encoding;
- **árboles**: no requieren escalado de las variables numéricas.

Después del One-Hot Encoding, los 27 predictores originales se representan mediante **77 columnas transformadas**.

La expansión de dimensionalidad es esperable: una variable categórica con varias categorías se representa mediante múltiples indicadores binarios.

## Control de leakage

La secuencia usada es:

    split train/test
        ↓
    ajustar transformaciones sólo en train
        ↓
    transformar train
        ↓
    aplicar esas mismas transformaciones a test

El conjunto de test no participa en el aprendizaje de medianas, modas, escalas ni categorías.

Cuando se realice validación cruzada, las transformaciones se mantendrán dentro de un `Pipeline`, de modo que cada fold aprenda el preprocesamiento sólo a partir de su propio subconjunto de entrenamiento.

## Decisiones derivadas del EDA

El EDA no se utiliza para eliminar predictores en esta etapa.

Los 27 predictores se conservan porque:

- una asociación descriptiva no equivale a importancia predictiva;
- la selección de variables antes del modelado podría introducir decisiones excesivamente guiadas por el target;
- la importancia de atributos se evaluará posteriormente como parte específica del TP.

Aunque la base actual no presenta valores faltantes, se mantiene la imputación dentro del pipeline por robustez y reproducibilidad metodológica.

## Conclusión del bloque

El preprocesamiento queda metodológicamente cerrado:

- partición 80/20 estratificada;
- prevalencia de Desertor estable entre train y test;
- transformaciones aprendidas únicamente desde train;
- variables categóricas codificadas mediante One-Hot;
- escalado aplicado cuando corresponde;
- 27 predictores originales → 77 columnas transformadas;
- ausencia de leakage hacia test;
- sin selección de variables basada en el EDA.

El siguiente bloque es la **clasificación binaria con Regresión Logística, SVM y Árbol de Decisión**.
