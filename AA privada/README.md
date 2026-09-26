# TP1: predicción con información del primer año

Esta carpeta permite leer dos respuestas a preguntas distintas sobre la misma base de
deserción estudiantil. El repositorio conserva su visibilidad **privada** y la primera resolución y su código están en `Resolución TP AA/`.

**Objetivo principal de la guía de la cátedra:** construir y comparar modelos
para predecir deserción con datos de inscripción y rendimiento del primer año.
Incluye una tarea binaria (`Desertor` frente al resto) y otra multiclase
(`Desertor`, `En Curso`, `Graduado`). **La guía no obliga a usar las 27
variables en todos los modelos**; ésa es la configuración completa que
escogimos para comparar algoritmos. La comparación por
etapas es una **pregunta adicional** sobre cuánta información añade cada
semestre; el TP no exige una alerta al ingreso.

## Empezá por acá

- [EDA explicado con los 27 gráficos](02_alternativa_temporal/eda/EDA_27_VARIABLES_COMENTADO.md): qué muestra cada columna y qué observar antes de modelar.
- [Mapa de asociación de las 27 variables](02_alternativa_temporal/eda/MAPA_ASOCIACIONES_27.md): cuáles muestran diferencias con el target, cifras y límites.
- [Hipótesis con datos que las motivan](02_alternativa_temporal/eda/README.md#hipótesis-reformuladas-observación--interpretación--prueba): cómo comprobarlas.
- [Primera resolución del TP](01_resolucion_original/README.md) y [alternativa por etapas](02_alternativa_temporal/README.md): las dos formas de abordar el trabajo.
- [Análisis conjunto de los resultados](COMPARACION_RESULTADOS.md): métricas, matrices de confusión, acuerdo entre informes y límites de la comparación.
- [Verificaciones de la cátedra](03_verificaciones_catedra/README.md): curvas train/CV, búsqueda de dos parámetros, 3 frente a 5 folds, Dummy, tamaños de muestra, OvR/OvO y funcionamiento de Random Forest.

[Guía original del TP](https://docs.google.com/document/d/1a7cHCdDvyGEcpK6Ek2F3Kir4JldNpJUJ/edit): requisitos orientativos del informe, clasificación, curvas, ensambles y atributos.

La carpeta se llama `AA privada` para encontrarla fácilmente; la **privacidad
la determina la configuración del repositorio**, no el nombre de la carpeta.

| Subcarpeta | Pregunta | Estado |
|---|---|---|
| `01_resolucion_original/` | ¿Cómo se comparan los algoritmos usando los 27 predictores? | Guía que enlaza al informe original y a su código canónico. |
| `02_alternativa_temporal/` | ¿Cómo cambia la predicción al incorporar información del primer y segundo semestre? | EDA sobre train, hipótesis, código reproducible y resultados exploratorios. |
| `03_verificaciones_catedra/` | ¿Qué pasa al variar complejidad, folds y tamaño de train? | Curvas, tablas por fold y explicación de OvR/OvO y Random Forest. |

## Qué se puede comparar

El informe original selecciona algoritmos y compara tareas binaria y multiclase.
El EDA de ambas resoluciones se recalculó sólo en las mismas 40.000 filas de
train, tras apartar 10.000 para test. Una versión anterior del EDA original
había visto las etiquetas de toda la base, por lo que las pruebas en este test
siguen siendo exploratorias y no confirman hipótesis de forma independiente.
La alternativa mantiene **un mismo clasificador por tarea** al comparar etapas; así,
el cambio entre etapas se debe a las columnas disponibles, dentro de las limitaciones
de la base y del procedimiento. No atribuimos una diferencia entre los dos informes
al algoritmo: las particiones y preguntas no son idénticas.

Las 19 variables que el informe original agrupa como «inscripción/contexto» no tienen
todas un momento de captura comprobado. En particular, matrícula, deuda, beca y
cuatro variables parentales se tratan como **bloque de fecha no verificada** en la
alternativa. Las doce restantes son candidatas a estar disponibles al ingreso; para
usarlas operativamente también habría que confirmar sus fechas con la fuente.

La base es sintéticamente aumentada y transformada. No contiene fecha individual de
abandono. Incluso si el modelo mejora con segundo semestre, eso no prueba que
hubiéramos podido intervenir *antes* del abandono.

## Orden de lectura

1. [`01_resolucion_original/README.md`](01_resolucion_original/README.md): qué hizo el TP previo.
2. [`02_alternativa_temporal/README.md`](02_alternativa_temporal/README.md): preguntas, disponibilidad y límites.
3. [`02_alternativa_temporal/eda/RECORRIDO_EDA.md`](02_alternativa_temporal/eda/RECORRIDO_EDA.md): preguntas y respuestas en escalera, del inventario a las dos vías de modelado.
4. [`02_alternativa_temporal/eda/EDA_27_VARIABLES_COMENTADO.md`](02_alternativa_temporal/eda/EDA_27_VARIABLES_COMENTADO.md): preguntas, lectura descriptiva y posibles comprobaciones para las 27 variables.
5. [`02_alternativa_temporal/eda/atlas/README.md`](02_alternativa_temporal/eda/atlas/README.md): los 27 gráficos y sus respuestas numéricas.
6. [`02_alternativa_temporal/eda/MAPA_ASOCIACIONES_27.md`](02_alternativa_temporal/eda/MAPA_ASOCIACIONES_27.md): todas las asociaciones observadas, los datos que las justifican y sus límites.
7. `02_alternativa_temporal/eda/README.md`: siete hipótesis y cómo contrastarlas; `TEORIAS_DESDE_EDA.md` desarrolla su lectura.
8. [`COMPARACION_RESULTADOS.md`](COMPARACION_RESULTADOS.md): comparación razonada de ambas resoluciones y errores por clase.
9. `02_alternativa_temporal/resultados_modelos/`: comparación numérica y matrices.
10. [`03_verificaciones_catedra/README.md`](03_verificaciones_catedra/README.md): comprobaciones adicionales para discutir las sugerencias de clase.

Los notebooks de la cátedra que motivan cada paso están vinculados en
`02_alternativa_temporal/CATEDRA_Y_DECISIONES.md`.
