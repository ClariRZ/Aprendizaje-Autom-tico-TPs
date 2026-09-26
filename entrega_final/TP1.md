# Predicción de deserción y trayectoria académica en educación superior politécnica

**Trabajo práctico 1 — Aprendizaje Automático, 2026**  
**Integrantes:** [completar]

## Resumen

Comparamos clasificadores para dos formulaciones de la trayectoria académica: Desertor frente a No desertor, y Desertor, En Curso y Graduado. Usamos 50.000 registros y 27 predictores de una versión sintéticamente aumentada de datos de estudiantes universitarios portugueses. Toda la exploración y la selección de hiperparámetros se hicieron en entrenamiento; como el test había sido observado en una exploración preliminar, sus cifras son exploratorias. En la tarea binaria, la Regresión Logística alcanzó ROC-AUC 0,9506 y un ensamble por votación suave, 0,9514. En tres clases, Random Forest obtuvo F1 macro 0,7808; En Curso fue la clase peor reconocida y su error dominante fue confundirse con Graduado, un error que la formulación binaria no puede registrar. Un análisis por bloques acumulativos mostró que el mayor salto de señal aparece al incorporar el primer semestre. Como los modelos completos usan información de todo el primer año y siete variables sin fecha de captura acreditada, su desempeño no equivale a capacidad de alerta temprana.

## 1. Introducción

La deserción universitaria no responde a un único factor: combina condiciones de ingreso, situación administrativa y económica, y rendimiento académico, que aparecen en momentos distintos de la trayectoria. Analizamos una base derivada de estudiantes del Instituto Politécnico de Portalegre, una institución pública portuguesa [1,2], con información de inscripción, variables administrativas y rendimiento durante el primer año.

Los trabajos sobre esta misma fuente muestran que la capacidad predictiva crece al incorporar información académica posterior al ingreso, sobre todo del primer semestre [2,3]. Eso tiene una consecuencia interpretativa: una variable puede predecir bien la deserción precisamente porque ya registra que la trayectoria se diferenció. Por eso no nos preguntamos sólo qué modelo rinde más, sino qué información sostiene la predicción y cuándo está disponible. El análisis es predictivo, no causal.

A partir del análisis exploratorio planteamos tres hipótesis:

- **H1.** La actividad y el rendimiento académico del primer año son los predictores más informativos de la trayectoria.
- **H2.** Las variables administrativas y las disponibles al ingreso aportan información, pero menos que el rendimiento académico.
- **H3.** En tres clases, En Curso tendrá menor sensibilidad y F1 que Desertor y Graduado; al agruparla con Graduado como No desertor, la evaluación binaria deja de contar los errores entre esos dos estados. H3 se debilita si En Curso obtiene métricas similares a las otras clases o si casi nunca se confunde con Graduado.

## 2. Materiales y métodos

### 2.1. Datos y variables explicativas

El conjunto proviene de los datos de entrenamiento de la competencia de Kaggle *Classification with an Academic Success Dataset* (Playground Series S4E6) [4], generados sintéticamente a partir del dataset de UCI *Predict Students’ Dropout and Academic Success* [2]. Sobre esa base se tradujeron y renombraron variables, se agruparon categorías, se reescalaron notas y se submuestreó de forma estratificada a 50.000 observaciones. Los resultados describen este conjunto transformado, no la cohorte original.

La base tiene 27 predictores (11 numéricos y 16 categóricos) y una respuesta. Los organizamos según su momento de captura (Figura 1): 12 candidatos de ingreso, cuya disponibilidad real al ingreso debería confirmarse con la fuente; cuatro del primer semestre (S1); cuatro del segundo (S2); y siete administrativos y familiares sin fecha de captura acreditada (deuda, matrícula al día, beca y cuatro parentales).

![Figura 1. Bloques de variables según su momento de captura](figuras/figura_01.svg)

**Figura 1.** Bloques de variables según su momento de captura. Las siete columnas sin fecha se usan en el modelo completo, pero no se les asigna disponibilidad temprana: el modelo de 27 variables no representa una predicción hecha al ingreso.

Primero se hizo una partición 80/20 (40.000 / 10.000) con semilla 42, estratificada por las tres clases. El análisis exploratorio (EDA) se calculó sólo sobre entrenamiento. Una exploración preliminar, anterior a fijar este protocolo, había observado las etiquetas de toda la base; por eso la evaluación en test no constituye una confirmación independiente, y la tratamos como exploratoria a lo largo del trabajo.

En entrenamiento, la respuesta se distribuyó en Graduado 18.966 (47,42 %), Desertor 13.223 (33,06 %) y En Curso 7.811 (19,53 %). No aplicamos remuestreo: preferimos conservar la distribución real y compensar el desbalance con la métrica (F1 macro y métricas por clase), que es donde se decide si la clase minoritaria pesa en la selección.

**Calidad de la muestra (sólo entrenamiento):**

- Hay 213 filas completas repetidas (respuesta incluida) y 235 repeticiones de los 27 predictores; en 22 perfiles idénticos (49 filas) la respuesta difiere. Sin identificador personal no se puede saber si son la misma persona, así que se conservaron y su efecto se midió con una sensibilidad sobre el test (Sección 3.7).
- En 34 casos de S1 y 62 de S2 hay más materias aprobadas que inscriptas (1 y 3 con aprobadas y cero inscriptas). Sin definiciones institucionales no hay criterio para corregirlos; se conservaron y documentaron.
- No hay faltantes explícitos, pero un cero en aprobadas o en nota puede codificar situaciones distintas (no rendir, desaprobar, abandonar). Por eso analizamos por separado la frecuencia de ceros y el nivel entre valores positivos, sin recodificar ceros como faltantes.
- Dos pares de columnas parentales son recodificaciones equivalentes; no las tratamos como evidencia independiente.

### 2.2. Tareas objetivo

En la tarea binaria, Desertor es la clase positiva y Graduado y En Curso forman No desertor. En la multiclase se conservan las tres etiquetas. En Curso es un estado registrado, no un desenlace: sin fecha ni duración de seguimiento no sabemos si esos estudiantes se graduarán o abandonarán. Para la tarea binaria se hizo otra partición 80/20 con la misma semilla, estratificada por la etiqueta binaria; ambos test tienen 10.000 registros, pero no exactamente las mismas filas, de modo que las métricas de una tarea no se comparan numéricamente con las de la otra.

### 2.3. Preprocesamiento, modelos y selección

Preprocesamiento y clasificador se integraron en un único *pipeline*, de modo que cada paso se reajusta sólo con los datos de entrenamiento de cada *fold* y nunca ve la validación. Imputamos numéricas por mediana y categóricas por moda (la base no tiene faltantes, pero el procedimiento queda preparado para registros nuevos incompletos) y codificamos las categóricas con One-Hot, lo que lleva los 27 predictores a 77 columnas. El EDA no se usó para descartar variables.

Elegimos clasificadores con sesgos inductivos distintos, para que ninguna conclusión dependa de una sola familia:

- **Regresión Logística:** modelo lineal en el logit; supone efectos aditivos y monótonos de cada predictor sobre el log-odds de desertar. C es la inversa de la penalización L2.
- **SVM con núcleo RBF:** busca la frontera de máximo margen en un espacio inducido por el núcleo; γ controla cuán local es la similitud y C cuánto se penalizan las violaciones del margen.
- **Árbol de decisión:** particiones recursivas por umbrales que minimizan la impureza de Gini; captura interacciones y no linealidades, pero tiene alta varianza si crece sin restricción.
- **K-NN:** clasifica por mayoría entre los k vecinos más cercanos; k chico da fronteras muy irregulares (bajo sesgo, alta varianza) y k grande las suaviza.
- **Random Forest:** promedia árboles entrenados sobre muestras bootstrap, eligiendo en cada división entre un subconjunto aleatorio de columnas. Si los árboles tienen varianza σ² y correlación ρ, la varianza del promedio es ρσ² + (1 − ρ)σ²/B [5]: agregar árboles reduce el segundo término, y el muestreo de columnas reduce ρ, que es lo que el promedio no puede eliminar. Por eso puede usar árboles profundos, de bajo sesgo, sin pagar su varianza. Fijamos `max_features="sqrt"`, es decir, 8 de las 77 columnas candidatas por división.
- **AdaBoost:** agrega árboles poco profundos en secuencia, reponderando los casos mal clasificados en cada ronda; reduce sobre todo el sesgo, y la tasa de aprendizaje regula cuánto aporta cada árbol.

Los hiperparámetros (Tabla 1) se eligieron por validación cruzada estratificada de tres folds sobre las 40.000 filas de entrenamiento; la configuración ganadora se reentrenó con todo el entrenamiento y se aplicó una sola vez al test. En binario el criterio fue ROC-AUC, que mide la capacidad de ordenar Desertor por encima de No desertor sin depender de un umbral. En tres clases fue F1 macro: el promedio simple del F1 de cada clase, que da a En Curso el mismo peso que a Graduado pese a tener menos de la mitad de registros.

**Tabla 1.** Grillas de hiperparámetros y valores elegidos por validación cruzada en entrenamiento.

| Tarea · modelo | Valores probados | Elegido |
|---|---|---|
| Binaria · Logística | C: 0,01; 0,1; 1; 10 | C=0,1 |
| Binaria · SVM | C: 0,1; 1; 10 · γ: scale; 0,01; 0,1 | C=0,1; γ=0,01 |
| Binaria · Árbol | profundidad: 2; 4; 6; 8; 12; sin límite · hoja mín.: 1; 10; 50 | prof. 8; hoja 50 |
| 3 clases · K-NN | k: 3; 5; 9; 15; 25 · peso: uniforme, distancia | k=25; uniforme |
| 3 clases · Árbol | profundidad: 2; 4; 6; 8; 12; sin límite · hoja mín.: 1; 10; 50 | prof. 12; hoja 50 |
| 3 clases · Random Forest | árboles: 200; 500 · profundidad: 12, sin límite · hoja mín.: 1; 50 | 500; sin límite; hoja 1 |
| 3 clases · AdaBoost | prof. base: 1; 2 · árboles: 100; 200; 400 · tasa: 0,05; 0,1 | prof. 2; 400; tasa 0,1 |

Además combinamos Logística, SVM y Árbol binarios por Soft Voting, promediando sus probabilidades con pesos iguales. Como el SVM no produce probabilidades, se calibró con una sigmoide (Platt) usando tres folds internos de entrenamiento (`CalibratedClassifierCV`, `ensemble=False`); ni la calibración ni los pesos usaron test. Las predicciones basadas en probabilidades asignaron la clase de mayor probabilidad (umbral 0,5 en binario) y el SVM individual, el signo de su función de decisión; la ROC-AUC se calculó siempre con puntajes continuos.

**Tres clases sin envolturas OvR/OvO.** Los cuatro clasificadores multiclase aprenden las tres etiquetas de forma nativa: en K-NN votan vecinos de las tres clases; en Árbol y Random Forest, Gini se calcula sobre las tres proporciones en cada nodo; en AdaBoost cada árbol base distingue las tres clases. Por eso no hizo falta descomponer el problema en tres clasificadores binarios uno-contra-resto (OvR) ni en tres por pares (OvO). OvR y OvO sí aparecen como dos formas de resumir la ROC-AUC multiclase a partir de las mismas probabilidades: OvR compara cada clase contra la unión de las otras dos; OvO promedia las AUC de cada par de clases [6].

**Comprobaciones complementarias (todas en entrenamiento):**

- **Curvas de complejidad:** variamos un hiperparámetro por vez y comparamos el puntaje en los folds de ajuste con el del fold reservado. Por costo computacional, las curvas de Random Forest y AdaBoost usaron 12.000 filas y la de K-NN 6.000; describen la forma de la curva, no reemplazan a los modelos finales. La de AdaBoost fijó profundidad 2 y tasa 0,1 y recorrió 100 a 600 árboles.
- **Ampliación de bordes:** como varias elecciones cayeron en el extremo de la grilla, extendimos K-NN (25, 35, 51 vecinos) y AdaBoost (200, 400, 600 árboles) con las 40.000 filas y tres folds. Esta extensión se hizo después de observar el test y no reeligió modelos.
- Una segunda grilla del Árbol de dos parámetros (12 combinaciones de profundidad y hoja mínima), una comparación de tres frente a cinco folds con parámetros fijos, el mismo Árbol con 10.000, 20.000 y 40.000 filas, y un clasificador Dummy que siempre predice la clase más frecuente.
- **Análisis por bloques acumulativos:** con el split estratificado por tres clases y folds idénticos en cada etapa, entrenamos Logística binaria y Árbol multiclase con 12 variables de ingreso, 16 (+S1), 20 (+S2) y, como escenario aparte, 27 (+7 sin fecha).
- **Importancia por permutación del bosque, sobre las 27 columnas originales:** se permutó cada columna completa antes del One-Hot, de modo que todas las categorías derivadas de una variable se alteran juntas, y se midió la caída del F1 macro en cada fold de validación (cinco repeticiones por columna y fold).

## 3. Resultados

### 3.1. Exploración inicial

**H1: información académica.** Dentro de Desertor, el 58,1 % tiene cero materias aprobadas en S1 y el 67,9 % en S2; en En Curso son 4,2 % y 4,6 %, y en Graduado, alrededor de 1,5 % en ambos semestres (Figura 2a). Entre quienes aprobaron al menos una materia, las medianas de S1 son 4/5/6 y las de S2, 3/4/6 (Desertor/En Curso/Graduado). La señal académica tiene entonces dos componentes: la ausencia de actividad (el cero), que separa a Desertor, y el nivel de rendimiento, que ordena a las otras dos. Un cero no implica haber rendido y desaprobado.

**H2: información administrativa y de ingreso.** La tasa de deserción fue 93,79 % con matrícula no al día (n=4.332), 76,05 % con deuda (n=2.877) y 7,21 % con beca (n=9.930), frente al 33,06 % general (Figura 2b). Son asociaciones muy fuertes, pero estas variables no tienen fecha de captura acreditada: la matrícula impaga podría ser consecuencia del abandono y no una señal previa. Entre los candidatos de ingreso, la mediana de edad fue 22 años en Desertor y 19 en las otras clases, mientras el puntaje de ingreso casi se solapa (medianas 61/61/63).

**H3: la clase En Curso.** Representa el 19,53 % del entrenamiento. Comparte con Graduado la mediana de edad (19 años), pero entre quienes aprobaron alguna materia tiene medianas menores (5 frente a 6 en S1; 4 frente a 6 en S2). Parece una clase intermedia en rendimiento, lo que anticipa confusión con Graduado. Pero eso es una conjetura del EDA: lo que la contrasta es la matriz de confusión del modelo multiclase.

![Figura 2a. Porcentaje con cero materias aprobadas por clase](figuras/figura_02a.svg)

*(a) Porcentaje con cero materias aprobadas por clase.*

![Figura 2b. Tasa de Desertor por grupo administrativo](figuras/figura_02b.svg)

*(b) Tasa de Desertor por grupo administrativo; la línea marca la tasa general.*

**Figura 2.** Contrastes exploratorios en entrenamiento (n=40.000) que motivan H1 y H2. Cada porcentaje usa como denominador su propio grupo, indicado en la etiqueta.

**Tabla 2.** Resultados binarios en el test (n=10.000). Precisión, sensibilidad y F1 se refieren a Desertor.

| Modelo | Accuracy | Precisión | Sensibilidad | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logística | 0,9066 | 0,9004 | 0,8067 | 0,8510 | 0,9506 |
| SVM | 0,8979 | 0,9150 | 0,7619 | 0,8315 | 0,9460 |
| Árbol | 0,9057 | 0,9064 | 0,7970 | 0,8482 | 0,9440 |
| Soft Voting | 0,9053 | 0,9080 | 0,7940 | 0,8472 | 0,9514 |

### 3.2. Clasificación binaria

La Logística obtuvo ROC-AUC 0,9506 y el mejor F1 (0,8510). El Soft Voting subió la AUC a 0,9514, una ganancia de 0,0008 sin mejora en F1 ni en sensibilidad. Es lo esperable cuando los modelos combinados dependen de las mismas variables académicas y cometen errores correlacionados: promediar sólo reduce la parte del error que no comparten. Que un modelo lineal empate con un SVM no lineal y con un árbol es, además, un resultado sobre los datos: sugiere que en la tarea binaria la relación entre rendimiento y deserción es predominantemente aditiva y monótona en el logit.

### 3.3. Complejidad y generalización

Una curva de complejidad aísla el efecto de un hiperparámetro: si el puntaje en entrenamiento sigue subiendo mientras el de validación se estanca o cae, el modelo empieza a ajustar ruido de la muestra.

**Modelos binarios (Figura 3).** La Logística es prácticamente insensible a C: la AUC de validación fue 0,9504 con C=0,01 y alrededor de 0,9507 con C entre 0,1 y 10. Con 40.000 filas y 77 columnas la penalización casi no restringe los coeficientes, y el modelo está limitado por su sesgo (linealidad) más que por la varianza. Esta curva usa el split estratificado por la etiqueta binaria; la comparación con el Dummy y de folds usa el split de tres clases y da 0,9503 (0,950 redondeado en la Figura 5): son validaciones sobre filas ligeramente distintas. En el SVM, la mejor región probada fue C=0,1 y γ=0,01; valores mayores de C empeoran la validación, un patrón de sobreajuste por margen demasiado estricto. El Árbol muestra el caso típico: con hoja mínima 50, la AUC de validación sube hasta 0,9445 en profundidad 8 y luego se aplana mientras la de entrenamiento sigue creciendo. Con profundidad 8, exigir 100 casos por hoja dio 0,9453 frente a 0,9445; la diferencia es pequeña, se observó después de la selección y no justificó reajustar el modelo.

![Figura 3. Curvas de complejidad binarias](figuras/figura_03.svg)

**Figura 3.** Curvas de complejidad binarias (gris: entrenamiento; rojo: validación, media de tres folds). Las escalas verticales difieren entre paneles.

**Modelos multiclase (Figura 4).** En Random Forest, la validación sube de 0,729 a 0,774 entre profundidades 4 y 12 y no mejora al quitar el límite, mientras el ajuste a entrenamiento sigue creciendo. Sin embargo, la validación no cae: es el efecto del promediado, que absorbe la varianza de los árboles profundos. En K-NN, k=3 sobreajusta (0,823 en entrenamiento frente a 0,700 en validación) y el máximo de esta curva está en k=11 (0,733). La búsqueda con 40.000 filas eligió k=25. No es contradictorio: con más datos aumenta la densidad de puntos, cada vecindario de k casos se vuelve más local, y el k óptimo crece con n. En AdaBoost, la validación sigue subiendo de 0,7656 con 100 árboles a 0,7771 con 600, sin caída en el rango probado. Es un comportamiento conocido del boosting con árboles poco profundos y tasa baja, que tarda en sobreajustar. Es decir, más complejidad no implica automáticamente sobreajuste: depende de cómo el algoritmo controla la varianza.

![Figura 4. Curvas de complejidad multiclase](figuras/figura_04.svg)

**Figura 4.** Curvas de complejidad multiclase (F1 macro; gris: entrenamiento; rojo: validación). El tamaño de muestra difiere por panel: se compara la forma de cada curva, no su altura.

### 3.4. Referencias y sensibilidad de las decisiones

**Dummy.** Predecir siempre la clase mayoritaria da AUC 0,500 en binario (frente a 0,950 de la Logística) y F1 macro 0,214 en tres clases (frente a 0,770 del Árbol de profundidad 12 y hoja 50) (Figura 5). El F1 macro del Dummy no es cero porque acierta todos los Graduado: su F1 en esa clase es alto, y en las otras dos, nulo.

**Número de folds.** Con cinco folds en lugar de tres y parámetros fijos, la AUC de la Logística fue 0,9505 frente a 0,9503 y el F1 macro del Árbol, 0,7690 frente a 0,7698. En el bosque (12.000 filas, 80 árboles), 0,7741 frente a 0,7726. Con folds de validación de 8.000 a 13.000 filas, la estimación ya es estable y aumentar el número de folds cambia poco.

**Segunda grilla del Árbol.** Con 40.000 filas, profundidad 12 y hoja 50 obtuvo F1 macro 0,7698; profundidad 8 y hoja 50, 0,7694. Un árbol sin límite y hoja 10 alcanzó 0,8250 en entrenamiento pero sólo 0,7446 en validación: la brecha es la firma del sobreajuste. Los 0,0004 entre las dos mejores configuraciones no son una diferencia práctica.

**Bordes de la grilla.** Los resultados de la ampliación están en la Tabla 3. En K-NN, ir más allá de 25 no mejora. En AdaBoost, 600 árboles supera a 400: el borde sigue abierto y el óptimo real podría estar más allá.

**Tamaño de muestra.** El mismo Árbol con 10.000, 20.000 y 40.000 filas dio 0,7729, 0,7672 y 0,7698.

![Figura 5. Referencia trivial y comparación de tres frente a cinco folds](figuras/figura_05.svg)

**Figura 5.** Referencia trivial y tres (gris) frente a cinco (rojo) folds, en entrenamiento.

**Tabla 3.** Ampliación de bordes: F1 macro en validación cruzada (40.000 filas, tres folds).

| Modelo (fijo) | Valor probado → F1 macro |
|---|---|
| K-NN (peso uniforme) | 25 → 0,7487 · 35 → 0,7475 · 51 → 0,7453 |
| AdaBoost (prof. 2; tasa 0,1) | 200 → 0,7623 · 400 → 0,7715 · 600 → 0,7747 |

No hay un orden creciente, pero como cada tamaño valida sobre casos distintos, esto no es una curva de aprendizaje: para serlo habría que fijar el conjunto de evaluación y repetir la extracción.

**AUC multiclase: OvR frente a OvO.** Para el Árbol de 40.000 filas, las mismas probabilidades dieron AUC macro 0,9159 resumida por OvR y 0,9071 por OvO. La diferencia es compatible con que OvO pondera por igual el par En Curso–Graduado, el más difícil, mientras que en OvR ese par queda diluido dentro de un “resto” que incluye a Desertor, fácil de separar. La selección siguió basándose en F1 macro.

### 3.5. Clasificación multiclase

Random Forest obtuvo el mayor F1 macro (0,7808) y AdaBoost, el F1 más alto para Desertor (0,8562). Los dos ensambles superan al Árbol individual por mecanismos opuestos: el bosque reduce la varianza promediando, y AdaBoost reduce el sesgo agregando árboles débiles.

**Tabla 4.** Resultados multiclase en el test (n=10.000). “Macro” pondera por igual las tres clases.

| Modelo | Accuracy | Precisión macro | Sensibilidad macro | F1 macro | ROC-AUC OvR |
|---|---:|---:|---:|---:|---:|
| K-NN | 0,7943 | 0,7664 | 0,7388 | 0,7483 | 0,9034 |
| Árbol | 0,8106 | 0,7774 | 0,7647 | 0,7699 | 0,9170 |
| Random Forest | 0,8196 | 0,7896 | 0,7750 | 0,7808 | 0,9254 |
| AdaBoost | 0,8182 | 0,7874 | 0,7707 | 0,7776 | 0,9219 |

H3 se sostiene en esta evaluación. En Curso tuvo el menor F1 en los cuatro modelos: 0,5542 (K-NN), 0,5899 (Árbol), 0,6096 (Random Forest) y 0,6006 (AdaBoost). En el bosque, su sensibilidad fue 0,5981, frente a 0,8076 en Desertor y 0,9192 en Graduado. La matriz de confusión (Figura 6) muestra hacia dónde van los errores: de los 1.953 registros reales En Curso, 569 (29,1 %) fueron clasificados como Graduado y 216 (11,1 %) como Desertor. La confusión con Graduado es la dominante, y es exactamente la que la formulación binaria no puede ver: esos 569 casos cuentan como aciertos cuando ambas clases se codifican No desertor. Esta lectura compara etiquetas; no atribuye al bosque un desempeño binario, porque los modelos binarios se entrenaron por separado.

![Figura 6. Matriz de confusión de Random Forest](figuras/figura_06.svg)

**Figura 6.** Matriz de confusión de Random Forest en test, normalizada por fila (estado real), con conteos. En rojo, los errores.

### 3.6. Cuándo aparece la señal

Con los mismos registros y folds en cada etapa (Figura 7), la AUC binaria de validación pasa de 0,7941 con las 12 variables de ingreso a 0,9262 al sumar S1 y a 0,9422 al sumar S2. En tres clases, el F1 macro del Árbol pasa de 0,5603 a 0,7115 y 0,7529. El salto grande está en S1: la mayor parte de la información que separa las trayectorias aparece en el primer semestre, y S2 agrega un refinamiento menor. Las siete variables sin fecha llevan los valores a 0,950 y 0,770, pero ese último tramo no corresponde a ningún momento temprano identificable.

### 3.7. Sensibilidad a perfiles repetidos entre entrenamiento y test

Recalculamos las métricas de los mismos modelos del análisis por bloques excluyendo del test 103 registros (1,03 %) cuyo vector de 27 predictores coincidía exactamente con alguna fila de entrenamiento (en 94 coincidía también la etiqueta). El modelo no se reentrenó; sólo cambió el conjunto evaluado.

**Tabla 5.** Métricas de test con y sin los 103 perfiles repetidos.

| Modelo | Métrica | Test completo (n=10.000) | Sin repetidos (n=9.897) |
|---|---|---:|---:|
| Logística binaria, 20 variables | ROC-AUC | 0,9446 | 0,9440 |
| Árbol multiclase, 20 variables | F1 macro | 0,7455 | 0,7448 |
| Logística binaria, 27 variables | ROC-AUC | 0,9518 | 0,9513 |
| Árbol multiclase, 27 variables | F1 macro | 0,7699 | 0,7694 |

![Figura 7. Puntajes por bloques acumulativos de variables](figuras/figura_07.svg)

**Figura 7.** Puntaje medio en tres folds de entrenamiento al acumular bloques de variables. El tramo rojo agrega las siete variables sin fecha de captura documentada.

Las diferencias (0,0005 a 0,0007) muestran que excluir coincidencias exactas cambia poco estas métricas en los modelos evaluados. Esto no demuestra ausencia de sesgo por perfiles repetidos. El filtro no detecta perfiles casi idénticos ni repeticiones en subconjuntos de variables, y no corrige la exposición previa del test.

### 3.8. Importancia de atributos

Antes de permutar, el bosque tenía F1 macro de 0,7818, 0,7821 y 0,7802 en los tres folds. La mayor caída fue la de materias aprobadas en S2 (0,1004), seguida por matrícula al día (0,0360) (Figura 8). Sumadas, las ocho variables académicas acumulan 0,1533; las tres administrativo-económicas, 0,0468; y las 16 de ingreso y contexto, 0,0075. Estas sumas son descriptivas y no una descomposición del F1: cuando dos predictores están correlacionados (por ejemplo, aprobadas y evaluaciones del mismo semestre), permutar uno deja al modelo usar el otro, y la importancia individual subestima la del bloque.

![Figura 8. Importancia por permutación en Random Forest](figuras/figura_08.svg)

**Figura 8.** Importancia por permutación en Random Forest: caída media del F1 macro en validación al permutar cada columna original.

## 4. Discusión

H1 queda respaldada por tres evidencias independientes en su método: el EDA (el cero académico concentra a Desertor), el análisis por bloques (el mayor salto aparece con S1) y la importancia por permutación (las variables académicas dominan). H2 se sostiene parcialmente: matrícula al día es la segunda variable más importante, pero su falta de fecha impide saber si antecede al abandono o lo registra; las variables de ingreso aportan poco (0,7941 de AUC en solitario, 0,0075 de importancia acumulada). H3 se confirma en los cuatro modelos, y la matriz de confusión identifica el mecanismo: En Curso se confunde sobre todo con Graduado, un error invisible para la tarea binaria. Sin seguimiento posterior, tampoco sabemos si parte de esa confusión refleja estudiantes que efectivamente se graduarán.

Sobre la elección de modelos, en binario la Logística iguala a alternativas más flexibles, lo que indica una estructura mayormente aditiva. En tres clases ganan los ensambles de árboles, que capturan mejor la frontera no lineal entre En Curso y Graduado. Las curvas de complejidad muestran que el sobreajuste depende del mecanismo de control de varianza de cada algoritmo, no sólo de su complejidad nominal: el Árbol sin límite sobreajusta, el bosque con árboles igual de profundos no, y AdaBoost todavía mejora con 600 árboles.

**Limitaciones.** La exploración preliminar observó las etiquetas del test, así que las cifras de test son exploratorias. Los datos están aumentados sintéticamente, incluyen filas idénticas y algunas inconsistencias, y En Curso no es un desenlace final. La grilla inicial terminó en bordes para SVM (menor C), K-NN (mayor k), Random Forest (más árboles, sin límite de profundidad, hoja mínima) y AdaBoost (máximos de árboles, profundidad y tasa). La ampliación descartó mejoras en K-NN, pero no en AdaBoost, y SVM y bosque no se exploraron fuera de sus bordes: los valores elegidos son los mejores entre los ensayados. Por último, los modelos completos usan información de todo el primer año y siete variables sin fecha: no demuestran capacidad de alerta temprana.

## 5. Conclusión

Con los 27 predictores del primer año, la Regresión Logística distinguió Desertor de No desertor con ROC-AUC 0,9506, y Random Forest clasificó los tres estados con F1 macro 0,7808. En Curso fue la clase más difícil y su error dominante, confundirse con Graduado, queda oculto en la formulación binaria. La mayor parte de la señal aparece con el rendimiento del primer semestre, lo que sugiere dónde podría ubicarse una alerta temprana. Para validarla hacen falta fechas individuales del abandono y de las variables administrativas, y una cohorte nueva e independiente.

## Bibliografía

1. Direção-Geral do Ensino Superior (DGES). *Sistema de Ensino Superior Português*. Ministério da Educação, Ciência e Inovação, Portugal. https://www.dges.gov.pt/pt/pagina/sistema-de-ensino-superior-portugues
2. Realinho V, Machado J, Baptista L, Martins MV. *Predicting Student Dropout and Academic Success*. Data. 2022;7(11):146. doi:10.3390/data7110146.
3. Martins MV, Baptista L, Machado J, Realinho V. *Multi-Class Phased Prediction of Academic Performance and Dropout in Higher Education*. Applied Sciences. 2023;13(8):4702. doi:10.3390/app13084702.
4. Reade W, Chow A. *Classification with an Academic Success Dataset*. Playground Series, Season 4, Episode 6. Kaggle; 2024. https://www.kaggle.com/competitions/playground-series-s4e6
5. Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2.ª ed. Springer; 2009. Cap. 15.
6. Hand DJ, Till RJ. *A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems*. Machine Learning. 2001;45:171–186.
