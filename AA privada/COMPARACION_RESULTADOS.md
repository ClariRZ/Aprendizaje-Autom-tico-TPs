# Qué muestran las dos resoluciones del TP1

**Objetivo de la guía del TP:** predecir la deserción mediante modelos binarios
y multiclase con información de inscripción y del **primer año**. No exige
literalmente las 27 columnas: usarlas juntas fue una decisión de la primera
resolución. La
[primera resolución](../Resoluci%C3%B3n%20TP%20AA/tp%20completo/TP1_completo.md) responde esa
consigna comparando algoritmos; la
[alternativa](02_alternativa_temporal/INFORME_ALTERNATIVO.md) agrega una
pregunta opcional: qué aporta cada bloque del año al mismo tipo de modelo.
Un número mayor de un informe frente al otro **no identifica el efecto del
algoritmo**, porque no siempre comparten partición y configuración. El EDA actual de ambas resoluciones se recalculó sólo sobre las 40.000 filas
de train con el mismo split multiclase. Una versión previa del EDA original
había examinado también las etiquetas de test. Por ese antecedente, ninguno de los
tests de esta base equivale a una validación confirmatoria nueva. Los dos
estudios usan la misma base de 50.000 filas, sintéticamente aumentada.

## 1. Qué es comparable

| Aspecto | Primera resolución | Alternativa por etapas | Consecuencia para la lectura |
|---|---|---|---|
| Pregunta | ¿Cómo se desempeñan Logística, SVM, Árbol y Voting en binario y K-NN, Árbol, Random Forest y AdaBoost en tres clases usando las **27 variables**? | ¿Qué cambia al añadir **S1** y luego **S2** al mismo tipo de modelo, y qué pasa con siete columnas de fecha incierta? | Comparar algoritmos **dentro de la primera**; comparar etapas **dentro de la segunda**. |
| Target | Binario: `Desertor` frente a `Graduado`+`En Curso`. Multiclase: tres estados. | Las mismas dos tareas. | ROC-AUC binaria y F1 macro multiclase responden preguntas diferentes; no comparar sus valores entre sí. |
| Entrenamiento/test | 40.000/10.000, semilla 42. En binario, split estratificado por target **binario**; en multiclase, por las **tres clases**. | 40.000/10.000, semilla 42, **un mismo split estratificado por tres clases** para ambas tareas. | El test binario no es exactamente el mismo. Diferencias pequeñas entre informes binarios no son ganancias demostradas. |
| Selección | `GridSearchCV`, 3 folds, ROC-AUC binario y F1 macro multiclase; familias y grillas propias del informe original. | `GridSearchCV`, 3 folds, mismas métricas, mismas filas/folds **entre etapas de una tarea**; Logística binaria y Árbol multiclase. | Las comparaciones por etapa estiman el aporte del **bloque dentro de ese protocolo**; las grillas y parámetros pueden cambiar entre etapas. Hay sensibilidad con parámetros fijos. |
| Momento de los datos | Todos los 27 juntos: incluye S1 y S2 y campos administrativos. | 12 candidatos de ingreso → 16 (+S1) → 20 (+S2); 7 campos sin fecha se añaden solo como sensibilidad: 19 → 23 → 27. | El caso de 27 columnas responde al objetivo de la guía; los pasos por semestre son una exploración adicional. |

[Protocolo y grupos exactos de columnas](02_alternativa_temporal/resultados_modelos/protocolo_ejecutado.json). El informe original agrupa 19 columnas como «inscripción/contexto», pero ese nombre no acredita la fecha de deuda, matrícula, beca ni de cuatro indicadores parentales. Incluso algunos de los 12 candidatos de ingreso necesitan confirmación de disponibilidad.

## 2. Primera resolución: qué algoritmos rindieron mejor con las 27

### Binario, test de 10.000 filas

| Modelo | ROC-AUC | F1 Desertor | Recall Desertor | Lectura |
|---|---:|---:|---:|---|
| Regresión Logística | **0,9506** | **0,8510** | **0,8067** | Mejor F1 y recall entre estos cuatro. |
| SVM | 0,9460 | 0,8315 | 0,7619 | Peor recall que Logística. |
| Árbol | 0,9440 | 0,8482 | 0,7970 | F1 cercano a Logística. |
| Soft Voting | **0,9514** | 0,8472 | 0,7940 | ROC-AUC supera Logística en **0,0008**; F1 y recall son menores. No defender superioridad práctica por esa sola diferencia. |

Fuente: [tabla original binaria](../Resoluci%C3%B3n%20TP%20AA/tp%20completo/graficos/02_tabla_modelos_binarios.svg) y [sección de resultados](../Resoluci%C3%B3n%20TP%20AA/tp%20completo/TP1_completo.md#clasificación-binaria). La selección de hiperparámetros por CV se hizo sobre train. En el Árbol original, aumentar profundidad después de la zona 6–8 elevó el ajuste en train sin mejorar la validación: señal de sobreajuste de configuraciones más complejas dentro de esta base.

### Tres clases, test de 10.000 filas

| Modelo | F1 macro | Accuracy | F1 `En Curso` | Lectura |
|---|---:|---:|---:|---|
| K-NN | 0,7483 | 0,7943 | 0,5542 | Menor F1 macro del grupo. |
| Árbol | 0,7699 | 0,8106 | 0,5899 | Referencia simple para la alternativa. |
| Random Forest | **0,7808** | **0,8196** | **0,6096** | Mejor F1 macro y mayor F1 `En Curso` de estas cuatro opciones. |
| AdaBoost | 0,7776 | 0,8182 | 0,6006 | F1 macro cercano a Random Forest; el informe original le atribuye el mayor F1 específico de Desertor (**0,8562**). |

Fuente: [tabla original multiclase](../Resoluci%C3%B3n%20TP%20AA/tp%20completo/graficos/03_tabla_modelos_multiclase.svg) y [métricas por clase del informe original](../Resoluci%C3%B3n%20TP%20AA/tp%20completo/TP1_completo.md#clasificación-multiclase-y-ensambles). El RF supera al Árbol completo en **0,0109** de F1 macro sobre la misma tarea del primer estudio, pero la diferencia con AdaBoost es **0,0032**: sin intervalos/repeticiones independientes, evitar afirmar una superioridad universal. `En Curso` sigue siendo la clase más difícil; incluso el mejor F1 observado allí es 0,6096.

**Qué utiliza el Random Forest:** la [importancia por permutación en CV de train](../Resoluci%C3%B3n%20TP%20AA/tp%20completo/graficos/04_tabla_importancia_variables.svg) cae **0,1004 de F1 macro** al permutar aprobadas S2, **0,0360** al permutar matrícula al día, **0,0226** con nota S2 y **0,0115** con aprobadas S1. Indica dependencia *de ese modelo entrenado*; columnas correlacionadas pueden compartir su señal. Son atributos del primer año pertinentes para la tarea completa, aunque matrícula al día no tiene fecha de captura acreditada. No son efectos causales.

## 3. Alternativa: qué añade cada bloque

Se compararon etapas con iguales folds e individuos dentro de cada tarea. **CV se usa para interpretar el contraste**; el test se muestra para entender errores y se considera exploratorio porque esta misma base ya había sido analizada en la primera resolución.

| Información que recibe el modelo | Variables | ROC-AUC binaria, CV | F1 macro tres clases, CV | Interpretación |
|---|---:|---:|---:|---|
| E0: candidatas al ingreso | 12 | 0,7941 | 0,5603 | Hay cierta información aun sin columnas académicas, pero no se han verificado todas las fechas. |
| E1: E0 + cuatro de S1 | 16 | 0,9262 | 0,7115 | Respecto de E0: **+0,1321** AUC y **+0,1512** F1 macro. Mayor incremento de la secuencia. |
| E2: E1 + cuatro de S2 | 20 | 0,9422 | 0,7529 | Respecto de E1: **+0,0160** AUC y **+0,0415** F1 macro. Añade información, aunque menos que S1. |
| E0+?: E0 + siete sin fecha | 19 | 0,8592 | 0,6064 | El bloque de siete añade señal **sin probar que existiera al ingreso**. |
| E1+?: E1 + siete sin fecha | 23 | 0,9396 | 0,7389 | El bloque de siete sigue asociado al resultado sobre S1. |
| E2+?: E2 + siete sin fecha | 27 | 0,9503 | 0,7698 | Es el caso completo con 27 para los modelos de esta alternativa. |

[Tabla completa de CV y parámetros](02_alternativa_temporal/resultados_modelos/metricas.csv). **Los cuatro campos agregados en E1 o E2 viajan juntos**: esta comparación no prueba que toda la ganancia corresponda sólo a materias aprobadas. El aumento por las siete columnas tampoco permite asignar la ganancia específicamente a deuda, matrícula o beca: incluye cuatro columnas familiares, con pares recodificados exactamente en train. La [sensibilidad con hiperparámetros fijos](02_alternativa_temporal/resultados_modelos/validacion_config_fija.csv) conserva la tendencia E0→E1→E2: AUC 0,7941→0,9262→0,9422 y F1 macro 0,5491→0,7109→0,7522. En cambio, añadir las siete columnas a E0 con Árbol de complejidad fija sólo aumenta **0,0134** el F1 macro medio y **reduce 0,0157** el score de uno de los tres folds. El incremento de ese bloque depende más del ajuste que el incremento por semestres. Son **tres folds**, no una estimación sólida de variabilidad entre cohortes.

### Los errores detrás de las métricas, en el mismo test alternativo

| Etapa | Desertor detectados / reales, binario | Desertor omitidos, binario | No desertor marcados como Desertor | F1 `En Curso`, tres clases |
|---|---:|---:|---:|---:|
| E0 | 1.640 / 3.306 | 1.666 | 670 | 0,3205 |
| E1 | 2.463 / 3.306 | 843 | 369 | 0,4733 |
| E2 | 2.593 / 3.306 | 713 | 313 | 0,5454 |
| E2+? | 2.682 / 3.306 | 624 | 284 | 0,5899 |

Fuente: [matrices de confusión](02_alternativa_temporal/resultados_modelos/matrices_confusion.csv), [métricas por clase](02_alternativa_temporal/resultados_modelos/metricas_por_clase.csv) y [test completo](02_alternativa_temporal/resultados_modelos/metricas.csv). De E0 a E1 se detectan **823 Desertor adicionales** en ese test y se reducen los falsos positivos en **301** con el umbral por defecto. De E1 a E2 son **130 adicionales** y **56 falsos positivos menos**. Es una comparación descriptiva sobre esas predicciones; no atribuye beneficio clínico/institucional ni demuestra que se hubiera podido intervenir a tiempo. Aun con las 27, aproximadamente **42 % de `En Curso`** no se clasifica como tal (recall 0,5771 en la alternativa completa).

## 4. ¿Coinciden las dos resoluciones?

**Sí en el caso verificable:** el Árbol multiclase con 27 predictores obtiene **F1 macro 0,7699** y **F1 `En Curso` 0,5899** tanto en la [primera resolución](../Resoluci%C3%B3n%20TP%20AA/tp%20completo/TP1_completo.md#clasificación-multiclase-y-ensambles) como en [E2+?](02_alternativa_temporal/resultados_modelos/metricas.csv). En ambos informes el Árbol completo usa profundidad **12** y mínimo de **50** por hoja; el mismo resultado funciona como comprobación de consistencia para ese caso. La primera resolución suma la comparación con Random Forest y AdaBoost, que la alternativa no repite por etapas.

**No usar una comparación impropia:** Logística binaria completa da ROC-AUC de test **0,9506** en la primera resolución y **0,9518** en E2+? de la alternativa. El split binario **no es exactamente igual**; la diferencia de **0,0012** no es evidencia de que una forma de resolver el TP sea mejor. Para conocer qué ocurre al añadir S1 y S2 usar la **comparación interna** de la alternativa, con la misma familia de modelo y los mismos folds.

## 5. Conclusiones sostenibles y lo que falta

1. Con las **27 columnas disponibles**, los modelos clásicos clasifican razonablemente esta base: Logística binaria ROC-AUC cercano a **0,95**, Random Forest multiclase F1 macro **0,7808**. **En Curso** conserva mayor confusión que las otras clases.
2. El avance del primer semestre agrega una diferencia de validación considerable a los candidatos de ingreso bajo la Logística y el Árbol elegidos; S2 agrega una diferencia menor. El EDA, las comparaciones de bloques y la importancia por permutación son **coherentes**, pero no permiten asignar la mejora de un bloque a una sola variable ni inferir causalidad.
3. La información administrativa y familiar sin fecha aumenta los scores en los escenarios completos. Su utilidad para un momento real de predicción exige **fecha de registro**, origen de las codificaciones y orden temporal respecto del abandono. S2 también puede registrar consecuencias de una baja ya ocurrida.
4. La cercanía entre CV y test muestra **concordancia interna en esta base**, no generalización a otra institución o cohorte. Hay **313 filas exactamente duplicadas** en las 50.000 y, en el split alternativo, **103 casos de test** tienen los 27 predictores idénticos a alguna fila de train. Quitarlos **sólo de la evaluación**, sin volver a entrenar, cambia AUC binaria E2 de **0,9446 a 0,9440** y F1 macro E2 de **0,7455 a 0,7448**; con 27 columnas las diferencias son también pequeñas (AUC 0,9518→0,9513; F1 macro 0,7699→0,7694). Este filtro identifica coincidencias en las 27 columnas: en E2 con 20 predictores pueden existir más perfiles repetidos. No demuestra independencia del resto. [Detalle y método](02_alternativa_temporal/INFORME_ALTERNATIVO.md#sensibilidad-a-coincidencias-exactas-con-train). Faltan identificador personal, fecha del abandono y validación externa.

**En términos de la guía:** la primera forma pregunta **cómo comparan los
algoritmos al clasificar con las 27 columnas del primer año completo**;
el análisis de las 27 variables es el resultado central. La alternativa
responde una pregunta adicional: **cómo cambia el rendimiento al incorporar
S1 y S2**. Si se quisiera convertir ese análisis adicional en una alerta
*anterior al abandono*, harían falta fechas que este CSV no aporta. Esa
exigencia adicional **no es una condición para responder el TP**.
