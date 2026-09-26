# EDA de la alternativa temporal: qué observamos y qué habría que probar

Esta carpeta documenta un análisis exploratorio del TP1 sobre **40.000 filas de entrenamiento**. Reservamos antes las otras 10.000 con `train_test_split(test_size=0.20, stratify=target, random_state=42)`. Los números, tablas e hipótesis siguientes proceden **sólo de train**. No seleccionamos hipótesis mirando test; test debe quedar reservado para el experimento completo. La base local de entrada no se copia a esta carpeta.

**Para empezar sin conocimientos previos:** leer [EDA en escalera](RECORRIDO_EDA.md),
después el [EDA comentado de las 27 variables](EDA_27_VARIABLES_COMENTADO.md),
que interpreta cada gráfico y propone qué comprobar a continuación, y el
[atlas de 27 variables](atlas/README.md), con los gráficos y las cifras.
El [mapa de las 27 asociaciones](MAPA_ASOCIACIONES_27.md) las ordena **dentro
de cada tipo**, señala la evidencia concreta y explica qué puede confundirse
con otra variable o depender de la fecha de medición.
Este README reúne las cifras, reservas y siete hipótesis para profundizar.

**Marco de la guía:** la tarea principal es comparar modelos para predecir
deserción con datos de inscripción y primer año. El caso completo de
nuestro estudio usa 27 variables; la guía no obliga a incluir todas. La separación de conjuntos por
semestre de esta alternativa responde una pregunta adicional sobre aporte
de información, sin exigir una predicción al ingreso.

## Pregunta y secuencia de trabajo

El target tiene tres estados: `Desertor`, `En Curso` y `Graduado`. Primero se define si se desea (a) identificar `Desertor` frente al resto o (b) separar los tres estados. Luego se comprueba qué se midió y en qué momento, se observan asociaciones y se formulan **hipótesis exploratorias**. Recién después se comparan modelos con validación dentro de train y se evalúa una vez en test. La matriz de confusión se calcula con las predicciones de un modelo: no es una propiedad inicial del EDA.

El análisis sigue la lógica didáctica de los Colab de la cátedra sobre EDA, validación y clasificación: primero describir variables y clases, después proponer comparaciones, luego entrenar/evaluar sin filtrar información de test. Los cortes de 0, 1–3 y 4 o más materias aprobadas de `09_tasas_estratificadas_train.csv` son **descriptivos y elegidos durante la exploración**; no son umbrales optimizados ni reglas validadas.

## Qué tipo tiene cada variable y cómo se analiza

El [diccionario de la base](../../../00_base_de_datos/diccionario_datos_TP1_2026.csv) describe **27 predictores**: 11 numéricos y 16 categóricos. `target` es una **categoría aparte** con tres valores (`Desertor`, `En Curso`, `Graduado`); no se cuenta como predictor. Los nombres siguientes coinciden exactamente con las columnas del CSV. La lista clasifica **tipo de dato**, no importancia predictiva ni fecha de disponibilidad.

| Grupo | Variables exactas | Cantidad | Primera lectura del EDA |
|---|---|---:|---|
| Numéricas: antecedentes e ingreso | `Cualificación_promedio_de_estudios_previos`, `Puntaje_en_examen_de_ingreso`, `Edad_al_momento_de_la_inscripción` | 3 | Rango, mediana, cuartiles y distribución total y por clase; buscar solapamiento entre clases, extremos y tamaños de grupo. |
| Numéricas: primer semestre | `Unidades_curriculares_1er_semestre_(inscrito)`, `Cantidad_de_evaluaciones_en_el_1er_semestre`, `Unidades_curriculares_1er_semestre_(aprobadas)`, `Nota_promedio_en_el_1er_semestre` | 4 | Por clase, separar **proporción con cero** de mediana y cuartiles **entre valores positivos**; revisar aprobadas > inscriptas. |
| Numéricas: segundo semestre | `Unidades_curriculares_2do_semestre_(inscrito)`, `Cantidad_de_evaluaciones_en_el_2do_semestre_`, `Unidades_curriculares_2do_semestre_(aprobadas)`, `Nota_promedio_en_el_2do_semestre` | 4 | Repetir lectura de ceros y positivos; comparar con S1 sin asumir que la deserción ocurrió después de medir S2. |
| Categóricas: antecedentes/contexto | `Estudios_máximos_antes_de_la_inscripción`, `estado_civil`, `modo_aplicacion`, `sexo`, `desplazado`, `Carrera`, `Asistencia_diurna/nocturna`, `Necesidades_educativas_especiales`, `Estudiante_Internacional` | 9 | Frecuencia de **cada** categoría, tasa de cada clase dentro de la categoría, `n` e intervalo; agrupar sólo para presentar si se documenta el criterio. |
| Categóricas: administrativas | `Tiene_deuda`, `Pago_de_matrícula_al_día`, `Poseedor_de_Beca` | 3 | Comparar `Sí` y `No`, cada uno con su denominador; verificar fecha de captura antes de incluirlos en un momento de predicción. |
| Categóricas **ordinales**: familia | `Demanda_Cog_Padre`, `Demanda_Cog_Madre`, `Ingresos_Padre`, `Ingresos_Madre` | 4 | Mostrar niveles en orden, `n` y tasas por nivel; revisar que cada par demanda/ingresos es una recodificación exacta observada en train. |

**Cómo usar esta clasificación, paso a paso:**

1. **Una variable sola (univariado):** para una numérica, mirar rango, mediana, cuartiles, ceros y distribución; para una categórica, contar `n` y porcentaje por nivel. Por ejemplo, `Tiene_deuda` es **categórica Sí/No**, aunque tenga sólo dos valores. Los ingresos parentales son categorías **ordenadas**, no montos numéricos para promediar.
2. **Variable × target (bivariado):** en numéricas, comparar distribución, ceros y positivos entre las tres clases; en categóricas, construir una tabla de contingencia y calcular la tasa de `Desertor` **dentro de cada categoría** con `n`. Ejemplo: `Tiene_deuda=Sí`, `n=2.877`, 76,05 % Desertor; `No`, `n=37.123`, 29,73 %. La asociación es descriptiva; no demuestra que la deuda precedió o causó la deserción.
3. **Predictores entre sí:** mirar correlaciones de rangos de las numéricas (`06_redundancia_train.csv`) y equivalencias entre categóricas. Dos variables casi idénticas pueden compartir su señal; no escoger una de ellas sólo por su asociación individual con target.
4. **Después del EDA:** formular una hipótesis y comparar dentro de train, con iguales folds y modelo, si añadir o quitar una variable o **bloque** cambia la validación. Las asociaciones bivariadas generan candidatas para estudiar: **no son un ranking de importancia multivariada**. La importancia individual puede ser inestable cuando las columnas están correlacionadas.

En `03_numericas_por_clase_train.csv` está el detalle numérico, con `n_positivos`; en `04_categoricas_por_clase_train.csv` están las tasas y tamaños de todas las categorías, incluso las pequeñas. `05_asociaciones_marginales_train.csv` resume η² para numéricas y V de Cramér para categóricas; **son medidas distintas, no valores intercambiables**. Para los grupos por semestre o estado de deuda, ver también `08_trayectorias_academicas_train.csv` y `09_tasas_estratificadas_train.csv`. Un cero en nota o aprobadas requiere confirmar su significado institucional; no hay valores faltantes `NaN`, pero eso no resuelve qué codifica cada cero.

## Datos, cobertura y principales hallazgos

### Cinco gráficos para leer las hipótesis

Los gráficos muestran **train** y usan rojo para señalar `Desertor` o una tasa elevada; los demás grupos se muestran en gris. Cada figura conserva su denominador visible o remite a la tabla detallada. Son una selección editorial del EDA: los resúmenes de las **27 variables** y todas las categorías permanecen en los CSV siguientes.

| Figura | Qué mirar | Pregunta que abre |
|---|---|---|
| [Clases](resultados/01_clases_train.png) | Tamaño de cada desenlace, con `n` y porcentaje. | ¿Debemos evaluar `En Curso` separadamente y usar F1 macro? |
| [Ceros académicos](resultados/02_ceros_academicos.png) | Proporción con cero aprobadas **dentro** de cada clase, en cada semestre. | ¿Qué agrega cada etapa y qué significa realmente el cero? |
| [Edad y puntaje de ingreso](resultados/03_puntajes_ingreso.png) | Mediana y rango intercuartílico; el puntaje se solapa entre las clases. | ¿Cuánto puede aportar la información al ingreso sin notas posteriores? |
| [Administrativas](resultados/04_categorias_financieras.png) | Tasa de Desertor entre cada categoría y `n`; la fecha de registro es desconocida. | ¿Estos indicadores existían antes del punto de predicción? |
| [Aprobaciones por semestre](resultados/05_trayectorias_academicas.png) | Tasa de Desertor para cuatro patrones entre inscriptos en ambos semestres. | ¿La diferencia entre etapas aporta señal, aun sin conocer fecha de abandono? |

El gráfico de edad y puntaje muestra el 25.º–75.º percentil, **no toda la distribución**; ver `03_numericas_por_clase_train.csv`. Los porcentajes de las figuras son asociaciones descriptivas, no efectos causales ni pruebas de anticipación.

Hay 27 predictores: 11 numéricos y 16 categóricos. En train: `Desertor` 13.223 (33,06 %), `En Curso` 7.811 (19,53 %) y `Graduado` 18.966 (47,42 %). Ninguna de las 27 columnas tiene faltantes codificados como `NaN`. Esto **no demuestra** que ceros o categorías sean datos completos: las calificaciones con cero deben revisarse en el diccionario antes de interpretarlas como nota obtenida.

| Observación en train | Cifras | Lectura provisional |
|---|---:|---|
| Cero en aprobadas de 1.er semestre, condicionado a clase | Desertor 58,15 %, En Curso 4,20 %, Graduado 1,47 % | La inactividad y/o falta de aprobación separa estados. No establece la causa ni el momento de la deserción. |
| Cero en aprobadas de 2.º semestre, condicionado a clase | 67,90 %, 4,62 % y 1,50 %, respectivamente | La asociación es todavía más marcada una vez observado el segundo semestre. |
| Mediana de aprobadas de 1.er semestre **entre positivos** | Desertor 4, En Curso 5, Graduado 6 | Incluso al excluir ceros quedan diferencias; son menores que la separación debida a ceros. |
| Sin aprobaciones con inscripción en ambos semestres | 6.697 estudiantes; 97,24 % Desertor, IC Wilson 95 % 96,82–97,60 % | Asociación muy grande, pero procede de información tardía y puede reflejar la definición del resultado. |
| Al menos una aprobación en ambos semestres | 30.157 estudiantes; 13,61 % Desertor, IC 13,23–14,00 % | La trayectoria académica concentra señal. No expresa un cambio de riesgo individual causado por aprobar. |
| Puntaje de ingreso, mediana | Desertor 61, En Curso 61, Graduado 63 | Solapamiento considerable de un indicador disponible al ingreso. |
| Edad de inscripción, mediana | Desertor 22, En Curso 19, Graduado 19 | Diferencia entre grupos; merece examinar distribución y contexto, no atribución causal. |

Las tasas condicionales de cero están calculadas **dentro de cada clase**. Son distintas de la probabilidad de ser desertor entre quienes tienen cero. Para esa pregunta mirar `08_trayectorias_academicas_train.csv`. La mediana positiva tiene su propio denominador (`n_positivos`): no mezclarla con la mediana que incluye ceros.

### Categóricas y tamaños de muestra

| Comparación en train | Tasa de Desertor | Precaución |
|---|---:|---|
| Matrícula al día: Sí (`n=35.668`) / No (`n=4.332`) | 25,68 % / 93,79 % | Se desconoce la **fecha** en que se registró el estado de pago. Podría ser posterior a la desvinculación. |
| Tiene deuda: No (`n=37.123`) / Sí (`n=2.877`) | 29,73 % / 76,05 % | Mismo problema de fecha y posible solapamiento conceptual con matrícula. |
| Beca: No (`n=30.070`) / Sí (`n=9.930`) | 41,59 % / 7,21 % | Fecha, criterios de asignación y continuidad desconocidos; no interpretar como efecto de la beca. |
| Acceso General (`n=27.723`) / Normativa Específica (`n=11.295`) | 22,72 % / 55,98 % | Modalidad y composición por carrera pueden mezclar diferencias. |
| Enfermería (`n=6.297`) / Animación y Diseño Multimedia (`n=1.474`) | 13,63 % / 71,30 % | Diferencia descriptiva entre carreras, no evaluación ajustada ni pronóstico transferible. |

En una categoría pequeña, la precisión disminuye: Tecnología en Producción de Biocombustibles tiene **38** casos en train, 73,68 % Desertor, IC Wilson 95 % 57,99–85,03 %. No ordenar carreras pequeñas como si sus porcentajes fueran exactos. El archivo categórico informa `n`, las tres clases y los intervalos para **cada** categoría, incluida la de grupos más pequeños.

Al estratificar por 4 o más aprobadas en 1.er semestre, la asociación descriptiva con matrícula permanece: No, `n=913`, 75,14 % Desertor; Sí, `n=25.930`, 8,59 %. Esto **no corrige** fecha de medición, confusión por carrera o variables socioeconómicas, selección ni causalidad. Permite decidir qué comparación hacer después y qué dato falta.

### Calidad, redundancia y ambigüedad de medición

- Hay **213** filas repetidas respecto de filas anteriores con target incluido y **235** repeticiones de predictores respecto de filas anteriores, dentro de train (la segunda cifra incluye las repeticiones exactas). Los predictores idénticos se asocian a targets diferentes en **22** patrones (49 filas). Sin identificador de estudiante no sabemos si son duplicaciones, trayectorias coincidentes o mediciones repetidas.
- En train, aprobadas supera inscritas en **34** registros del 1.er semestre y **62** del 2.º; en **1** y **3** filas, respectivamente, se registran aprobadas con cero inscritas. Puede deberse a definiciones institucionales distintas; hay que contrastar diccionario y fuente antes de borrar registros.
- `Demanda_Cog_Padre` y `Ingresos_Padre` se determinan mutuamente de forma exacta en train; ocurre lo mismo para las columnas de madre. Es redundancia **observada en esta base**, no prueba de que cognición e ingresos sean equivalentes en el mundo real. Revisar si hubo recodificación o error en generación de datos antes de dar una interpretación sustantiva.
- Inscritas 1.er/2.º semestre muestran correlación de rangos **0,979**; aprobadas 1.er/2.º, **0,921**. La importancia individual de variables correlacionadas puede repartirse entre ellas. Estas correlaciones no hacen redundantes sus fechas: una medición posterior puede añadir información útil que antes no existía.
- La asociación marginal numérica con las tres clases, medida como **η² en train**, es 0,529 para aprobadas 1.er semestre y 0,613 para aprobadas 2.º semestre; la categórica de matrícula al día tiene **V de Cramér 0,451**. η² y V **no se comparan entre sí**; ninguna es importancia multivariada validada ni efecto causal.

## Hipótesis reformuladas: observación → interpretación → prueba

Reformulamos las hipótesis a partir de los contrastes **más legibles** del EDA: avance académico en S1 y S2, situación económica/administrativa, características de ingreso y redundancia. Todos los números siguientes proceden de **40.000 filas de train**; hay 13.223 `Desertor`, 7.811 `En Curso` y 18.966 `Graduado`. Las cifras son asociaciones exploratorias: una hipótesis que se formuló mirando estos datos no queda «demostrada» porque coincida con ellos. Los modelos ya ejecutados se reportan [por separado](../INFORME_ALTERNATIVO.md) y esta base ya se usó en el TP original: test no es una cohorte totalmente independiente para validar las ideas nuevas.

**Regla de denominadores:** «58,1 % de Desertor tiene cero aprobadas S1» = 7.689/13.223, mientras que «92,7 % de quienes tienen cero aprobadas S1 es Desertor» = 7.689/8.296. No intercambiar estas dos preguntas. Las tasas por categorías, como «93,8 % de matrícula no al día es Desertor», usan el tamaño **de esa categoría** como denominador.

| Hipótesis de trabajo | Datos observados en train que la motivan | Prueba y resultado que la debilitaría |
|---|---|---|
| **H1. El progreso del primer semestre añade información a los datos de ingreso.** | Cero aprobadas S1: **7.689/13.223 Desertor (58,1 %)**, **328/7.811 En Curso (4,2 %)**, **279/18.966 Graduado (1,5 %)**. Entre quienes tienen al menos una aprobada, mediana **4, 5 y 6**, respectivamente. La nota S1 tiene casi los mismos ceros: posible solapamiento. [Tabla numérica](resultados/03_numericas_por_clase_train.csv); [figura](atlas/figuras/06.svg). | Comparar **ingreso → ingreso+S1** con mismas filas, folds y familia de modelo; medir F1 macro y recall por clase. Si no mejora de forma consistente, cae la propuesta de aporte adicional. Confirmar antes qué significan cero aprobadas y nota cero; una asociación puede reflejar abandono anterior al cierre S1. |
| **H2. El segundo semestre añade información sobre la que ya aporta S1.** | Cero aprobadas S2: **8.978/13.223 Desertor (67,9 %)**, **361/7.811 En Curso (4,6 %)**, **284/18.966 Graduado (1,5 %)**. Entre positivos, mediana **3, 4 y 6**. [Tabla numérica](resultados/03_numericas_por_clase_train.csv); [figura](atlas/figuras/10.svg). | Comparar **ingreso+S1 → ingreso+S1+S2** con mismos folds y algoritmo. Si no mejora, S2 puede estar repitiendo información. Incluso si mejora, **no** demuestra alerta previa: faltan fechas individuales de abandono. |
| **H3. La situación económica/administrativa se asocia con el estado, pero su utilidad temporal depende de cuándo se registró.** | Matrícula **No** al día: **4.063/4.332 Desertor (93,8 %)**; **Sí**: **9.160/35.668 (25,7 %)**. Deuda **Sí**: **2.188/2.877 (76,1 %)**; **No**: **11.035/37.123 (29,7 %)**. Beca **Sí**: **716/9.930 (7,2 %)**; **No**: **12.507/30.070 (41,6 %)**. [Tabla categórica](resultados/04_categoricas_por_clase_train.csv); [figuras](atlas/README.md#categóricas-administrativas-3). | Averiguar **fecha de captura, definición, vigencia y relación con el abandono** antes de usarlas para predecir en un momento determinado. Comparar estratos con suficiente tamaño y, si son previas, ablar el bloque en CV. Si son posteriores o dependen de la deserción, un score alto no prueba utilidad anticipatoria; tampoco un efecto causal de deuda o beca. |
| **H4. Existe información en el perfil de ingreso, además de la evolución académica.** | Edad mediana **22** en Desertor frente a **19** en En Curso y Graduado. Carrera: **858/6.297 (13,6 %) Desertor en Enfermería** frente a **1.051/1.474 (71,3 %) en Animación y Diseño Multimedia**. Acceso General **6.299/27.723 (22,7 %)** frente a Normativa Específica **6.323/11.295 (56,0 %)**. [Edad](atlas/figuras/03.svg); [carrera](atlas/figuras/17.svg); [admisión](atlas/figuras/14.svg). | Evaluar CV con ingreso frente a un clasificador trivial; mirar edad **dentro de** carrera y modalidad y calibración por subgrupo. Si asociaciones cambian al estratificar o aportan poco al modelo, no atribuirles una explicación individual. |
| **H5. Dos pares familiares repiten una misma codificación en esta base.** | `Demanda_Cog_Padre`↔`Ingresos_Padre` y `Demanda_Cog_Madre`↔`Ingresos_Madre` son correspondencias **bidireccionales exactas en train**. [Redundancias](resultados/06_redundancia_train.csv). | Verificar diccionario/generación; comparar CV con ambas y con una columna de cada par. Si no hay pérdida consistente, evitar interpretar dos importancias independientes. No concluir equivalencia real entre cognición e ingresos. Su fecha de captura también es desconocida. |
| **H6. Hace falta evaluar `En Curso` como tercera clase.** | `En Curso` es **7.811/40.000 (19,5 %)**. En Ingeniería Informática representa **35,6 %** (n=1.588 en la carrera), mientras que `Graduado` representa **5,0 %**. [Target](resultados/01_clases_train.csv); [carrera](resultados/04_categoricas_por_clase_train.csv). | Informar matriz de confusión, precision/recall/F1 por clase y F1 macro. Un binario Desertor/resto puede ocultar fallos para `En Curso`. Confirmar cuánto tiempo se siguió a cada estudiante antes de dar una interpretación definitiva a esa etiqueta. |
| **H7. Resumir la transición S1→S2 podría facilitar al modelo detectar perfiles.** | Entre estudiantes con inscripción en ambos semestres, **1+ aprobadas S1 → 0 S2**: **n=1.545, 92,4 % Desertor**; **1+ → 1+**: **n=30.157, 13,6 % Desertor**. [Trayectorias con denominadores](resultados/08_trayectorias_academicas_train.csv). | Comparar el **mismo modelo** con los conteos originales y con un indicador derivado de transición usando CV. Si mejora, es por la **representación**: no agrega datos nuevos. Tampoco revela si la deserción ocurrió antes o después de S2. |

**Conclusión que sí permite el EDA:** en esta base, las mayores diferencias visibles están en la **actividad/progreso académico** y en la **situación administrativa**; también hay diferencias por edad, carrera y admisión. Las notas y las aprobadas pueden solaparse entre sí; los indicadores administrativos requieren fecha para una predicción por etapas. [η² para numéricas y V de Cramér para categóricas](resultados/05_asociaciones_marginales_train.csv) son medidas distintas, y ninguna mide por sí sola aporte adicional o causalidad. Las siete columnas de fecha desconocida **siguen siendo parte de las 27 variables** de la opción completa.

## Qué necesitamos confirmar antes de un experimento «temprano»

1. **Momento de la foto y del resultado.** ¿Cuándo se asignó `Desertor`, `En Curso` o `Graduado`? ¿Cuánto seguimiento tuvieron los estudiantes? Sin esto no se demuestra predicción anticipada ni se distingue censura de una categoría estable.
2. **Fecha de disponibilidad por columna.** Siete variables parecen ligadas al ingreso por nombre o descripción: estudios previos, modalidad de admisión, carrera, calificación previa, examen de ingreso, edad al inscribirse y `estado_civil` (el diccionario dice expresamente «al momento de inscripción»). Otras cinco (`sexo`, `desplazado`, asistencia, necesidades educativas y condición internacional) son **candidatas** a existir al ingreso, pero su fecha debe confirmarse. Los cuatro indicadores por semestre sólo se usan al cierre de su semestre. Las siete restantes, tres administrativas (`Tiene_deuda`, `Pago_de_matrícula_al_día`, `Poseedor_de_Beca`) y cuatro parentales, tienen fecha de captura **desconocida**. El inventario no convierte candidatos en disponibilidad demostrada.
3. **Significado de cero.** Confirmar si nota 0 y aprobadas 0 indican desempeño observado, falta de exámenes o ausencia de cursada. Separar «cero» de «missing» cuando corresponda.
4. **Unidad del registro.** Solicitar ID de estudiante/cohorte/período; duplicados y filas con predictores idénticos pueden romper la independencia de la partición por filas. Con grupos reales, reevaluar la estrategia de split.
5. **Pertinencia de la cohorte.** Una única base sin fechas ni centro impide validación temporal o por institución. La comparación de bloques de columnas simula **disponibilidad por etapa**, no una validación hacia el futuro.

## Archivos y reproducción

`eda_train.py` lee un CSV sin modificarlo, aparta test al comienzo y genera en `resultados/`: manifiesto de fuente y split; distribución de clases; inventario de 27 variables; resúmenes numéricos con ceros/mediana de positivos; tasas categóricas e IC Wilson; asociaciones marginales separadas por métrica; pares redundantes; chequeos; trayectorias de semestres; tasas estratificadas; cinco gráficos PNG. `atlas_variables.py` genera el [atlas](atlas/README.md) con **27 gráficos SVG**, uno por predictor, y controla que fuente y partición de train sean las mismas. Los nombres de tablas tienen prefijo `01`–`09` y todos se refieren a train. No guardar en Git el CSV original ni las etiquetas o predicciones de test por fila.

```bash
python -m pip install pandas numpy scipy scikit-learn matplotlib
python eda_train.py --input /ruta/privada/dataset_TP1_2026.csv
python atlas_variables.py --input /ruta/privada/dataset_TP1_2026.csv
```

El `source_sha256` del manifiesto identifica exactamente la versión de datos analizada. El script exige las tres etiquetas conocidas y falla si cambia el target. Los intervalos Wilson describen incertidumbre de proporciones muestrales; no incorporan sesgo de selección, cambios de cohorte ni multiplicidad de exploraciones.
