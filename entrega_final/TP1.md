# Predicción de la deserción estudiantil: comparación de modelos de clasificación en una base sintética

**Trabajo práctico 1 — Aprendizaje Automático, 2026**  
**Integrantes:** [completar]

## Resumen

El objetivo de este trabajo fue comparar modelos de clasificación para predecir la deserción estudiantil, evaluar su desempeño y capacidad de generalización. Se analizó una base sintética transformada de 50.000 registros y 27 predictores mediante clasificación binaria —Desertores frente a No desertores— y multiclase —Desertores, En Curso y Graduados—.

Los hiperparámetros se seleccionaron mediante validación cruzada en entrenamiento. La comparación principal utilizó todos los predictores, incluidos los de ambos semestres. Como análisis adicional, se evaluó el aporte de incorporar sucesivamente información de inscripción y antecedentes, primer semestre, segundo semestre y variables administrativas y familiares.

En los modelos binarios, Soft Voting obtuvo la mayor ROC-AUC (0,9514), apenas por encima de regresión logística (0,9506), sin mejorar su F1. En los modelos multiclase, Random Forest alcanzó el mayor F1 macro (0,7808), mientras que En Curso fue la clase más difícil y se confundió principalmente con Graduado. La mayor mejora predictiva apareció al incorporar el primer semestre.

Los resultados muestran capacidad predictiva exploratoria en esta base. La exploración previa del test limita la evaluación, y el uso de información de ambos semestres no demuestra anticipación del abandono desde la inscripción.

## 1. Introducción

La predicción de la deserción estudiantil mediante aprendizaje automático permite evaluar la utilidad de los registros institucionales para reconocer estudiantes en riesgo [1]. Este trabajo examina esa capacidad predictiva mediante dos tareas: distinguir Desertor de No desertor y diferenciar los estados Desertor, En Curso y Graduado. Se compararon modelos de clasificación y se analizó el aporte de la información académica, administrativa y de ingreso.

Los antecedentes destacan la relevancia del rendimiento académico y de las condiciones financieras. Delen [1], a partir de ocho años de datos institucionales, identificó ambos grupos entre los predictores más importantes. En el Instituto Politécnico de Portalegre, Portugal, Realinho et al. [2] encontraron cinco predictores relevantes en los cuatro algoritmos examinados: materias aprobadas de ambos semestres, nota del segundo semestre, carrera y matrícula al día. Sobre esta fuente, Martins et al. [3] compararon distintas etapas del primer año: el mayor F1 global reportado pasó de 0,650 con información de inscripción a 0,745 al incorporar el primer semestre.

Estos antecedentes motivaron el análisis de una versión sintética y transformada de esa fuente [4]. El objetivo fue comparar el desempeño y examinar la capacidad de generalización de los modelos, así como identificar qué información contribuyó a sus predicciones. Se utilizaron todos los predictores y, como análisis adicional, se incorporaron grupos de variables de forma acumulativa. A partir de los antecedentes y de la exploración en entrenamiento se formularon tres hipótesis exploratorias:

- La actividad y el rendimiento académico del primer año aportan la mayor información predictiva **(H1)**.
- Las variables administrativas y de ingreso aportan información, pero menos que las académicas **(H2)**.
- En Curso presenta menor sensibilidad y F1 que las otras clases y se confunde principalmente con Graduado; agrupar ambas como No desertor oculta esa confusión **(H3)**.

## 2. Materiales y métodos

### 2.1. Datos y variables explicativas

Se utilizó el archivo `dataset_TP1_2026.csv`, con 50.000 registros, 27 predictores y una variable respuesta. La base procede de Kaggle Playground Series S4E6 [4], un conjunto sintético derivado de *Predict Students’ Dropout and Academic Success* [2]. El script de preparación disponible [5] documenta traducciones, agrupamientos, reescalados y muestreo por respuesta y carrera. El presente análisis utilizó el archivo ya preparado, cuyos nombres y categorías están en castellano.

Los 11 predictores numéricos y 16 categóricos se organizaron en cuatro grupos (Figura 1):

- **Inscripción y antecedentes:** 12 variables, entre ellas edad al inscribirse, carrera y estudios previos.
- **Primer semestre (S1):** materias inscriptas, evaluaciones, materias aprobadas y nota promedio.
- **Segundo semestre (S2):** las mismas cuatro medidas correspondientes a ese período.
- **Administrativas y familiares:** deuda, matrícula al día, beca y cuatro variables derivadas de ocupaciones parentales. Se desconoce la fecha de registro o actualización de este grupo. Los rótulos de ingresos y demanda cognitiva son recodificaciones de ocupación.

![Figura 1. Grupos de variables y análisis acumulativo](figuras/figura_01.svg)

**Figura 1. Grupos de predictores.** La comparación principal incluyó las 27 variables. El análisis acumulativo agregó grupos a partir de las 12 variables de inscripción y antecedentes; el grupo administrativo y familiar no representa una etapa posterior a S2.

El análisis exploratorio se realizó en el entrenamiento de la partición multiclase: 18.966 registros Graduado (47,42 %), 13.223 Desertor (33,06 %) y 7.811 En Curso (19,53 %). El desbalance se consideró en la evaluación mediante F1 macro y métricas por clase; no se aplicaron remuestreo ni pesos de clase al entrenamiento.

En este conjunto no se detectaron faltantes explícitos. Se conservaron 213 repeticiones de filas completas y 235 repeticiones de perfiles de predictores; 22 perfiles presentaban etiquetas diferentes. También se conservaron 34 casos de S1 y 62 de S2 con más materias aprobadas que inscriptas, por falta de información para corregirlos. Dos pares de variables parentales eran recodificaciones equivalentes. Los ceros académicos se analizaron por separado y no se trataron como faltantes.

### 2.2. Variable respuesta y particiones

La tarea binaria definió Desertor como clase positiva (1) y agrupó En Curso y Graduado como No desertor (0). La tarea multiclase conservó las tres etiquetas. En Curso representa un estado observado, sin información sobre la trayectoria posterior.

Cada tarea utilizó una partición estratificada por su propia respuesta, con 80 % para entrenamiento (40.000 registros), 20 % para prueba (10.000) y semilla 42. Las particiones binarias y multiclase no contienen exactamente los mismos registros. Una exploración previa había examinado las etiquetas de toda la base; por ello, el conjunto de prueba no constituye una evaluación independiente y sus resultados se consideran exploratorios.

### 2.3. Preprocesamiento y modelos

El análisis se implementó en Python con scikit-learn. La comparación principal incluyó los 27 predictores. El preprocesamiento y el clasificador se integraron en un *pipeline*, ajustado dentro de cada partición de entrenamiento de la validación cruzada. Se incluyeron imputadores por mediana y moda, que no reemplazaron valores por ausencia de faltantes. La codificación *one-hot* de las variables categóricas, junto con las numéricas, produjo 77 columnas. Las numéricas se estandarizaron para regresión logística, SVM y K-NN. Se conservaron todos los predictores.

En la tarea binaria se compararon regresión logística con penalización L2, máquinas de soporte vectorial con núcleo RBF (SVM) y árboles de decisión con criterio Gini. La regresión logística transforma una combinación lineal de predictores en una probabilidad mediante la función logística; L2 limita la magnitud de sus coeficientes. SVM RBF admite fronteras no lineales, mientras que el árbol representa interacciones mediante divisiones sucesivas. Su complejidad se controló mediante los hiperparámetros de la Tabla 1.

En multiclase se evaluaron K-NN, árboles de decisión, Random Forest (RF) y AdaBoost. K-NN determina la clase a partir de los vecinos más próximos. RF combina árboles entrenados con muestras *bootstrap* y selección aleatoria de variables en cada división, una estrategia orientada a reducir la varianza [6]; se fijó `max_features="sqrt"`. AdaBoost combina árboles pequeños de forma secuencial, aumentando el peso de los casos mal clasificados. Estos modelos admiten directamente las tres clases.

### 2.4. Selección de hiperparámetros y evaluación

Para cada modelo se definió una grilla de valores candidatos (Tabla 1). Todas sus combinaciones se evaluaron mediante validación cruzada estratificada de tres particiones (*folds*) dentro de entrenamiento. En cada iteración se utilizaron aproximadamente 26.667 registros para ajustar y 13.333 para validar. Se eligió la configuración con mayor ROC-AUC media en binario o mayor F1 macro medio en multiclase; luego se reajustó con los 40.000 registros y se evaluó en prueba.

**Tabla 1. Hiperparámetros evaluados y configuraciones seleccionadas mediante validación cruzada.**

| Tarea · modelo | Valores probados | Elegido |
|---|---|---|
| Binaria · Logística | C: 0,01; 0,1; 1; 10 | C=0,1 |
| Binaria · SVM | C: 0,1; 1; 10 · γ: scale; 0,01; 0,1 | C=0,1; γ=0,01 |
| Binaria · Árbol | profundidad: 2; 4; 6; 8; 12; sin límite · hoja mín.: 1; 10; 50 | prof. 8; hoja 50 |
| 3 clases · K-NN | k: 3; 5; 9; 15; 25 · peso: uniforme, distancia | k=25; uniforme |
| 3 clases · Árbol | profundidad: 2; 4; 6; 8; 12; sin límite · hoja mín.: 1; 10; 50 | prof. 12; hoja 50 |
| 3 clases · Random Forest | árboles: 200; 500 · profundidad: 12, sin límite · hoja mín.: 1; 50 | 500; sin límite; hoja 1 |
| 3 clases · AdaBoost | prof. base: 1; 2 · árboles: 100; 200; 400 · tasa: 0,05; 0,1 | prof. 2; 400; tasa 0,1 |

C es la inversa de la intensidad de regularización; γ, el parámetro del núcleo RBF; k, la cantidad de vecinos; hoja mín., el mínimo de casos por hoja.

Se construyó un ensamble Soft Voting con los tres modelos binarios seleccionados, promediando sus probabilidades con pesos iguales. Las probabilidades del SVM se obtuvieron mediante calibración sigmoide del pipeline completo con `CalibratedClassifierCV`, tres folds internos y `ensemble=False`. Esta validación interna se utilizó para calibrar probabilidades y fue distinta de la búsqueda de hiperparámetros.

Se calcularon accuracy, precisión, sensibilidad (*recall*), F1 y ROC-AUC. F1 combina precisión y sensibilidad; ROC-AUC evalúa la discriminación entre clases a distintos umbrales. En binario, precisión, sensibilidad y F1 se calcularon para Desertor. En multiclase se utilizaron promedios macro, que asignan igual peso a cada clase al evaluar; la AUC se calculó comparando cada clase frente al resto (OvR) y promediando sus valores. Las predicciones probabilísticas asignaron la clase más probable; el SVM individual utilizó el signo de su función de decisión.

### 2.5. Análisis complementarios

El aporte de los grupos de variables se estudió mediante regresión logística binaria y árbol multiclase. Se incorporaron sucesivamente inscripción y antecedentes (12 predictores), S1 (16), S2 (20) y variables administrativas y familiares (27). Se mantuvieron las filas y los folds dentro de cada tarea, seleccionando hiperparámetros en cada etapa. La comparación evalúa el desempeño alcanzado con cada conjunto de información y no aísla el efecto de agregar variables a una configuración fija. En ambas tareas se utilizó la partición estratificada por tres clases, distinta de la partición binaria principal.

Se construyeron curvas de complejidad comparando entrenamiento y validación al variar un hiperparámetro. Por costo computacional, las curvas de RF y AdaBoost utilizaron 12.000 registros y la de K-NN, 6.000. Se incluyó una referencia Dummy que predice la clase mayoritaria, se compararon tres y cinco folds con configuraciones fijas y se realizó una segunda búsqueda para el árbol. Las ampliaciones de las grillas de K-NN y AdaBoost se efectuaron después de observar el conjunto de prueba y no modificaron los modelos finales.

La importancia de variables se estimó mediante la caída del F1 macro de RF al permutar cada predictor original antes de *one-hot*, con cinco repeticiones por variable en cada uno de tres folds de validación. Finalmente, se recalcularon las métricas del análisis acumulativo excluyendo del conjunto de prueba los perfiles idénticos a los de entrenamiento, sin reajustar los modelos.

## 3. Resultados

### 3.1. Características de las clases

La proporción de registros sin materias aprobadas fue mayor en Desertor: 58,1 % en S1 y 67,9 % en S2, frente a 4,2 % y 4,6 % en En Curso y aproximadamente 1,5 % en Graduado para ambos semestres (Figura 2a). Entre quienes aprobaron al menos una materia, las medianas para Desertor, En Curso y Graduado fueron, respectivamente, 4, 5 y 6 en S1, y 3, 4 y 6 en S2. La ausencia de aprobaciones no permite distinguir entre no rendir y rendir sin aprobar.

La proporción de Desertor fue 93,79 % entre quienes no tenían la matrícula al día (n=4.332), 76,05 % entre quienes tenían deuda (n=2.877) y 7,21 % entre quienes tenían beca (n=9.930), frente al 33,06 % del entrenamiento completo (Figura 2b). La mediana de edad fue 22 años en Desertor y 19 en las otras clases.

![Figura 2a. Porcentaje con cero materias aprobadas por clase](figuras/figura_02a.svg)

*(a) Porcentaje con cero materias aprobadas dentro de cada clase.*

![Figura 2b. Proporción de Desertor por grupo administrativo](figuras/figura_02b.svg)

*(b) Proporción de Desertor dentro de cada grupo administrativo. Rojo: Desertor; gris: En Curso o Graduado. Cada barra suma 100 %; n indica el tamaño del grupo.*

**Figura 2. Asociaciones exploratorias en entrenamiento (n=40.000).** Cada porcentaje utiliza como denominador su propio grupo.

### 3.2. Clasificación binaria

La regresión logística obtuvo el mayor F1 (0,8510). Soft Voting alcanzó la mayor ROC-AUC (0,9514), frente a 0,9506 de regresión logística, pero presentó menor sensibilidad y F1 (Tabla 2).

**Tabla 2. Desempeño de los modelos binarios en prueba (n=10.000).**

| Modelo | Accuracy | Precisión | Sensibilidad | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logística | 0,9066 | 0,9004 | 0,8067 | 0,8510 | 0,9506 |
| SVM | 0,8979 | 0,9150 | 0,7619 | 0,8315 | 0,9460 |
| Árbol | 0,9057 | 0,9064 | 0,7970 | 0,8482 | 0,9440 |
| Soft Voting | 0,9053 | 0,9080 | 0,7940 | 0,8472 | 0,9514 |

Precisión, sensibilidad y F1 corresponden a Desertor. Accuracy representa la proporción total de aciertos; precisión, la proporción de desertores entre los casos señalados; sensibilidad, la proporción detectada entre los desertores reales. ROC-AUC: área bajo la curva ROC.

### 3.3. Clasificación multiclase

RF obtuvo el mayor F1 macro (0,7808), seguido por AdaBoost (0,7776), árbol (0,7699) y K-NN (0,7483) (Tabla 3).

**Tabla 3. Desempeño de los modelos multiclase en prueba (n=10.000).**

| Modelo | Accuracy | Precisión macro | Sensibilidad macro | F1 macro | ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| K-NN | 0,7943 | 0,7664 | 0,7388 | 0,7483 | 0,9034 |
| Árbol | 0,8106 | 0,7774 | 0,7647 | 0,7699 | 0,9170 |
| Random Forest | 0,8196 | 0,7896 | 0,7750 | 0,7808 | 0,9254 |
| AdaBoost | 0,8182 | 0,7874 | 0,7707 | 0,7776 | 0,9219 |

Los promedios macro asignan igual peso a las tres clases. ROC-AUC OvR corresponde al promedio macro de las comparaciones de cada clase frente al resto.

En Curso presentó el menor F1 en los cuatro modelos: 0,5542 en K-NN, 0,5899 en árbol, 0,6096 en RF y 0,6006 en AdaBoost. En RF, su sensibilidad fue 0,5981, frente a 0,8076 para Desertor y 0,9192 para Graduado.

De los 1.953 registros En Curso, RF clasificó 569 (29,1 %) como Graduado y 216 (11,1 %) como Desertor (Figura 3). La primera confusión queda oculta al agrupar En Curso y Graduado en una única categoría. Esta propiedad de las etiquetas no equivale al resultado de entrenar un clasificador binario.

![Figura 3. Matriz de confusión de Random Forest](figuras/figura_06.svg)

**Figura 3. Matriz de confusión de RF en prueba.** Cada fila presenta los conteos y porcentajes de una clase real; los errores aparecen en rojo.

### 3.4. Complejidad y generalización

En la tarea binaria, la AUC de validación de regresión logística varió poco entre C=0,01 y 10 (aproximadamente 0,9504–0,9507). En SVM, aumentar C desde la mejor región ensayada redujo el desempeño de validación. En el árbol con hoja mínima de 50 registros, la AUC alcanzó 0,9445 en profundidad 8 y luego se estabilizó, mientras el desempeño de entrenamiento continuó aumentando (Figura 4).

![Figura 4. Curvas de complejidad binarias](figuras/figura_03.svg)

**Figura 4. Curvas de complejidad de los modelos binarios.** AUC media de entrenamiento (gris) y validación (rojo) en tres folds sobre 40.000 registros. Las escalas verticales difieren entre paneles.

En multiclase, el RF reducido aumentó su F1 macro de validación de 0,7294 a 0,7741 entre profundidades 4 y 12; sin límite alcanzó 0,7738. Esta curva utilizó 80 árboles y hoja mínima de 10 registros, una configuración distinta de la final. K-NN con k=3 obtuvo 0,823 en entrenamiento y 0,700 en validación; el mayor valor de la curva se observó en k=11. La búsqueda principal seleccionó k=25 con otra muestra y otros candidatos. AdaBoost aumentó de 0,7656 a 0,7771 entre 100 y 600 árboles, sin descenso en el rango evaluado (Figura 5).

![Figura 5. Curvas de complejidad multiclase](figuras/figura_04.svg)

**Figura 5. Curvas de complejidad de los modelos multiclase.** F1 macro medio de entrenamiento (gris) y validación (rojo) en tres folds. Árbol: 40.000 registros; RF y AdaBoost: 12.000; K-NN: 6.000. Cada curva corresponde a su propio protocolo.

### 3.5. Referencia y sensibilidad de la selección

La referencia Dummy obtuvo ROC-AUC 0,500 en binario y F1 macro 0,214 en multiclase, por debajo de regresión logística y árbol (Figura 6).

Con configuraciones fijas, el cambio de tres a cinco folds modificó la AUC de regresión logística de 0,9503 a 0,9505 y el F1 macro del árbol de 0,7698 a 0,7690. En RF reducido, con 80 árboles, profundidad 12 y hoja mínima 10, cambió de 0,7741 a 0,7726. Esta comparación utilizó la partición estratificada por tres clases y no repitió la búsqueda completa de hiperparámetros.

![Figura 6. Referencia Dummy y comparación de folds](figuras/figura_05.svg)

**Figura 6. Referencia Dummy y sensibilidad al número de folds.** Desempeño de configuraciones fijas con tres folds (gris) y cinco (rojo) dentro de entrenamiento. Los paneles utilizan métricas diferentes.

En la segunda grilla del árbol, profundidad 12 y hoja mínima 50 alcanzó F1 macro de validación 0,7698, frente a 0,7694 con profundidad 8 y la misma hoja. Sin límite de profundidad y con hoja mínima 10, el F1 macro fue 0,8250 en entrenamiento y 0,7446 en validación.

La ampliación de las grillas no mejoró K-NN más allá de 25 vecinos, mientras que AdaBoost continuó mejorando hasta 600 árboles (Tabla 4). Estas comprobaciones posteriores no modificaron las configuraciones finales.

**Tabla 4. Ampliación de grillas: F1 macro medio en validación de tres folds sobre 40.000 registros.**

| Modelo | Configuración fija | Hiperparámetro | Valor | F1 macro |
|---|---|---|---:|---:|
| K-NN | pesos uniformes | *k* (n.º de vecinos) | 25 | **0,7487** |
| | | | 35 | 0,7475 |
| | | | 51 | 0,7453 |
| AdaBoost | profundidad 2; tasa 0,1 | n.º de árboles | 200 | 0,7623 |
| | | | 400 | 0,7715 |
| | | | 600 | **0,7747** |

La negrita indica el mayor F1 macro entre los valores probados para cada modelo.

### 3.6. Aporte acumulativo de los grupos de variables

La incorporación de S1 a las 12 variables de inscripción y antecedentes aumentó la AUC de validación de regresión logística de 0,7941 a 0,9262. Al agregar S2 se alcanzó 0,9422 y, con las variables administrativas y familiares, 0,9503. En el árbol multiclase, los valores de F1 macro fueron 0,5603, 0,7115, 0,7529 y 0,7698, respectivamente (Figura 7). En ambas tareas, el mayor incremento se observó al incorporar S1.

![Figura 7. Desempeño al agregar grupos de variables](figuras/figura_07.svg)

**Figura 7. Desempeño en validación cruzada del análisis acumulativo.** Se incorporaron sucesivamente 12, 16, 20 y 27 predictores, manteniendo las filas y los folds dentro de cada tarea y seleccionando hiperparámetros en cada etapa. El punto rojo incorpora el grupo administrativo y familiar, cuya fecha de disponibilidad es desconocida.

### 3.7. Importancia de los predictores

Las mayores caídas del F1 macro de RF por permutación correspondieron a materias aprobadas en S2 (0,1004) y matrícula al día (0,0360) (Figura 8). Las sumas descriptivas de importancias positivas fueron 0,1533 para las ocho variables académicas, 0,0468 para deuda, matrícula y beca, y 0,0075 para las otras 16 variables. Estas sumas no equivalen a permutar grupos completos ni descomponen el desempeño del modelo.

![Figura 8. Importancia por permutación en Random Forest](figuras/figura_08.svg)

**Figura 8. Importancia por permutación en RF.** Caída media del F1 macro de validación al permutar cada predictor original. Un valor mayor indica mayor dependencia predictiva del modelo respecto de esa variable.

### 3.8. Sensibilidad a los perfiles repetidos

En el análisis acumulativo, 103 registros de prueba compartían los mismos 27 predictores con alguna fila de entrenamiento; en 94 también coincidía la etiqueta. Al excluirlos sin reajustar los modelos, las métricas disminuyeron entre 0,0005 y 0,0007 (Tabla 5).

**Tabla 5. Desempeño de los modelos del análisis acumulativo con y sin perfiles repetidos en prueba.**

| Modelo | Métrica | Test completo (n=10.000) | Sin repetidos (n=9.897) |
|---|---|---:|---:|
| Logística binaria, 20 variables | ROC-AUC | 0,9446 | 0,9440 |
| Árbol multiclase, 20 variables | F1 macro | 0,7455 | 0,7448 |
| Logística binaria, 27 variables | ROC-AUC | 0,9518 | 0,9513 |
| Árbol multiclase, 27 variables | F1 macro | 0,7699 | 0,7694 |

Las métricas se recalcularon sin reentrenamiento. La AUC 0,9518 corresponde a la partición del análisis acumulativo, distinta de la utilizada en la Tabla 2.

## 4. Discusión

Los resultados brindaron respaldo exploratorio a la hipótesis de que la actividad y el rendimiento académico del primer año aportarían la mayor información predictiva (H1). Las diferencias entre clases en materias aprobadas, la mejora al incorporar S1 y la importancia por permutación fueron consistentes con esa expectativa. La relevancia académica coincide con Realinho et al. [2] y con la mejora al incorporar información del primer semestre descrita por Martins et al. [3]. Los puntajes no son directamente comparables por las diferencias entre bases y procedimientos.

La hipótesis de un aporte administrativo y de ingreso menor que el académico (H2) recibió respaldo parcial. El grupo administrativo y familiar mejoró el desempeño y matrícula al día fue el segundo predictor más importante, en consonancia con la relevancia financiera señalada por Delen [1] y Realinho et al. [2]. Sin embargo, el orden de incorporación de los grupos y la correlación entre predictores limitan una jerarquía general de sus aportes. Además, las fechas desconocidas de los datos administrativos impiden establecer si precedían al abandono.

La mayor dificultad para reconocer En Curso y su confusión con Graduado fueron consistentes con la tercera hipótesis (H3). En Curso obtuvo el menor F1 en los cuatro modelos y, en RF, la menor sensibilidad. La clasificación binaria oculta los errores entre En Curso y Graduado al reunir ambos estados. El análisis descriptivo, la comparación acumulativa y la importancia por permutación ofrecen evidencia complementaria sobre las hipótesis, aunque provienen de la misma base.

La elección del modelo depende de la métrica: en binario, regresión logística obtuvo el mayor F1 y Voting la mayor AUC; en multiclase, RF y AdaBoost alcanzaron los mayores F1 macro. Las diferencias observadas no establecen superioridad estadística. Las curvas muestran que aumentar la flexibilidad puede mejorar el ajuste sin mejorar la validación, en consonancia con el control de complejidad y varianza [6]. AdaBoost aún mejoró al ampliar su grilla, por lo que las configuraciones seleccionadas deben interpretarse dentro de los rangos ensayados.

El origen sintético, la exploración previa del conjunto de prueba, las inconsistencias conservadas y los perfiles repetidos limitan la generalización. Excluir coincidencias exactas tuvo un efecto pequeño, pero no descarta sesgos por perfiles similares. La importancia por permutación tampoco identifica causas y puede verse afectada por predictores correlacionados. Finalmente, utilizar información de ambos semestres permite reconocer dificultades ya presentes; una aplicación temprana requeriría datos disponibles antes del abandono, validación independiente y umbrales elegidos según el costo de los errores.

## 5. Conclusión

Los modelos permitieron distinguir los estados académicos con capacidad predictiva exploratoria en esta base sintética. El rendimiento académico tuvo un aporte destacado y el mayor incremento de desempeño se observó al incorporar el primer semestre. Soft Voting alcanzó la mayor AUC binaria (0,9514), regresión logística el mayor F1 binario (0,8510) y RF el mayor F1 macro multiclase (0,7808). La dificultad para reconocer En Curso mostró la información que se pierde al agruparlo con Graduado. Estos resultados sustentan la comparación realizada, pero su aplicación como alerta temprana requiere evaluación independiente con datos disponibles antes del abandono.

## Bibliografía

1. Delen D. *Predicting Student Attrition with Data Mining Methods*. Journal of College Student Retention: Research, Theory & Practice. 2011;13(1):17–35. doi:10.2190/CS.13.1.b.
2. Realinho V, Machado J, Baptista L, Martins MV. *Predicting Student Dropout and Academic Success*. Data. 2022;7(11):146. doi:10.3390/data7110146.
3. Martins MV, Baptista L, Machado J, Realinho V. *Multi-Class Phased Prediction of Academic Performance and Dropout in Higher Education*. Applied Sciences. 2023;13(8):4702. doi:10.3390/app13084702.
4. Reade W, Chow A. *Classification with an Academic Success Dataset*. Playground Series, Season 4, Episode 6. Kaggle; 2024. https://www.kaggle.com/competitions/playground-series-s4e6
5. *generar_dataset_tp1.py*. Script de preparación disponible para este TP, sin autoría ni fecha consignadas. [Código fuente](../00_base_de_datos/generar_dataset_tp1.py).
6. Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2.ª ed. Springer; 2009. Cap. 15.
