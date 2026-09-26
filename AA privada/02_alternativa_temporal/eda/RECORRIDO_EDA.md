# EDA en escalera: primero entender, después decidir qué modelar

Este recorrido responde preguntas sencillas **antes de mirar qué modelo gana**.
Se basa en 40.000 filas de train (split estratificado 80/20, semilla 42) de la
misma base de 50.000 estudiantes que utiliza el TP. El
[EDA comentado](EDA_27_VARIABLES_COMENTADO.md) guía la lectura de las
27 columnas y el atlas presenta
[una figura y una respuesta para cada variable](atlas/README.md).
Los [cinco gráficos de síntesis](README.md#cinco-gráficos-para-leer-las-hipótesis)
son la versión breve para una presentación; el atlas sirve para revisar cada
variable. Todas las comparaciones con el target se calcularon **en train**.

## Peldaño 0. ¿Qué queremos explicar o predecir?

La pregunta principal de la guía es **predecir deserción con datos de
inscripción y primer año** mediante clasificación binaria y multiclase.
Nuestro caso completo utiliza las 27 columnas, sin que la guía lo exija. La
comparación ingreso → S1 → S2 es una extensión para comprender qué aporta
cada bloque; no cambia lo que exige la consigna.

El target tiene tres categorías: **Graduado**, **En Curso** y **Desertor**.
En train hay respectivamente 18.966, 7.811 y 13.223 estudiantes.
Las dos tareas posibles son: `Desertor` frente a las otras dos clases juntas
(binaria), y distinguir las tres clases (multiclase). La segunda conserva
la diferencia entre `En Curso` y `Graduado`. [Ver el gráfico de tamaños](resultados/01_clases_train.png).

**Pregunta básica:** ¿Por qué no alcanza con decir «el modelo acertó el 80 %»?
**Respuesta:** las clases tienen tamaños distintos y `En Curso` puede ser
difícil de reconocer. Más adelante compararemos F1 por clase, F1 macro y
matrices de confusión, además del ROC-AUC de la tarea binaria. Estas métricas
requieren predicciones de modelos: aún no son parte del EDA.

## Peldaño 1. ¿Qué contiene cada columna por sí sola?

El [diccionario](../../../00_base_de_datos/diccionario_datos_TP1_2026.csv)
y el [inventario de train](resultados/02_inventario_y_calidad_train.csv)
identifican **11 numéricas** y **16 categóricas**. El
[README del EDA](README.md#qué-tipo-tiene-cada-variable-y-cómo-se-analiza)
las enumera una por una. En el atlas se grafica **cada** variable.

| Pregunta | Recurso | Respuesta de esta base |
|---|---|---|
| ¿Qué edades, puntajes y recuentos hay? | Histograma, rango, mediana y cuartiles. | Edad 17–70; puntaje de ingreso 48–95. La nota de estudios previos también toma 48–95 en train. |
| ¿Cuántas personas hay en cada categoría? | Barras de frecuencia con `n`; nunca mirar una tasa sin denominador. | Por ejemplo, `Tiene_deuda=Sí` aparece en 2.877 y `No` en 37.123 registros. |
| ¿Hay ceros o faltantes? | `NaN`, porcentaje cero y significado del diccionario. | Las 27 columnas no tienen `NaN`; esto **no prueba** que un cero en nota equivalga a una evaluación rendida. |
| ¿Hay incoherencias visibles? | Comprobar reglas entre columnas. | En train hay 34 casos con más aprobadas que inscriptas en S1 y 62 en S2; documentarlos sin borrarlos arbitrariamente. |

Una variable **numérica** requiere distribución: una mediana sola oculta
valores extremos y ceros. Una **categórica** requiere frecuencia por nivel:
`Carrera` tiene 15 categorías y algunas son pequeñas. Las cuatro variables
familiares son **categorías ordenadas**, no pesos ni importes numéricos.

## Peldaño 2. ¿Cómo se relaciona cada variable con el target?

Esto es el **EDA bivariado**: una variable × las tres clases. Los gráficos del
atlas ponen a la izquierda la distribución global, y a la derecha la
comparación con el target. Los archivos
[`03_numericas_por_clase_train.csv`](resultados/03_numericas_por_clase_train.csv)
y [`04_categoricas_por_clase_train.csv`](resultados/04_categoricas_por_clase_train.csv)
contienen todas las cifras, incluidos tamaños, ceros e intervalos.

### Una numérica: edad al inscribirse

**Pregunta:** ¿La distribución de edad cambia entre estados?
**Respuesta:** la mediana es 22 años en Desertor y 19 en En Curso y Graduado.
[El gráfico de edad](atlas/figuras/03.svg) deja ver el rango intercuartílico,
para no convertir una diferencia de medianas en una regla que clasifica a
cada estudiante. Edad pasa a ser **una variable que vale la pena probar junto
con las demás**; todavía no sabemos su aporte predictivo exclusivo.

### Una numérica con muchos ceros: aprobadas en primer semestre

**Pregunta A:** entre *los desertores*, ¿qué porcentaje tiene cero aprobadas
en S1? **58,15 %**. Su denominador son los 13.223 desertores.

**Pregunta B (distinta):** entre *quienes tienen cero aprobadas en S1*,
¿qué porcentaje está etiquetado Desertor? **92,68 %** de 8.296 estudiantes;
entre quienes tienen una o más son **17,46 %** de 31.704.

[Ver aprobadas S1](atlas/figuras/06.svg) y
[comparación S1/S2](resultados/02_ceros_academicos.png). Al mirar sólo los
valores positivos, la mediana de materias aprobadas en S1 es 4, 5 y 6 para
Desertor, En Curso y Graduado. Por eso separamos **tener cero** de **cuánto
se aprobó entre los que aprobaron**. Estas son asociaciones observadas;
las aprobaciones pueden reflejar un abandono ya ocurrido.

### Una categórica: deuda Sí/No

**Pregunta:** ¿difiere la tasa de Desertor entre niveles?
**Respuesta:** 76,05 % de los 2.877 con deuda y 29,73 % de los 37.123 sin
deuda. [Gráfico de deuda](atlas/figuras/21.svg).
Es una diferencia llamativa. **Antes** de incluir deuda en una supuesta
predicción al ingreso hace falta saber cuándo se registró: el diccionario
no contiene esa fecha. Tampoco se puede inferir que cancelar la deuda
cambie la probabilidad de deserción.

### Una categórica con grupos de tamaño desigual: carrera

**Pregunta:** ¿las tasas por carrera son iguales? En train, Enfermería tiene
13,63 % de Desertor (n=6.297); Animación y Diseño Multimedia, 71,30 %
(n=1.474). [Ver las 15 carreras](atlas/figuras/17.svg).
Tecnología en Producción de Biocombustibles tiene una tasa de 73,68 %
pero sólo **38** casos: por su tamaño no conviene presentarla como una
conclusión estable. La diferencia entre carreras no dice qué ocurre
manteniendo constantes edad, modalidad o rendimiento.

## Peldaño 3. ¿Hay columnas que repiten información?

**Pregunta:** ¿vale lo mismo observar una variable aislada que agregarla
cuando otras variables parecidas ya están en el modelo?
**Respuesta:** no necesariamente. Aprobadas S1/S2 tienen correlación de
rangos 0,921; inscriptas S1/S2, 0,979. La información se solapa, aunque
las columnas pueden corresponder a momentos distintos.
Las parejas `Demanda_Cog_Padre`/`Ingresos_Padre` y la pareja equivalente
para la madre tienen correspondencia exacta entre categorías en train.
[Ver redundancias](resultados/06_redundancia_train.csv).

Las medidas de [asociación marginal](resultados/05_asociaciones_marginales_train.csv)
ayudan a ordenar **por separado** numéricas (η²) y categóricas
(V de Cramér). No se pueden comparar ambos números como si fueran una misma
escala, ni representan la contribución de una columna cuando están todas juntas.

## Peldaño 4. ¿Qué hipótesis salen de lo observado?

| Observación | Hipótesis razonable | Qué resultado la debilitaría |
|---|---|---|
| Aprobadas S1 y S2 difieren mucho por clase. | Añadir indicadores del primer semestre mejora la clasificación sobre información de ingreso; S2 podría agregar algo más. | Que el score de validación no cambie con las mismas filas, folds y familia de modelo. |
| Edad y puntaje ya existen al ingreso y muestran diferencias, aunque los puntajes se solapan. | Podrían aportar cierta clasificación usando sólo información disponible en ese momento. | Que un modelo de entrada no supere un baseline o que añadirlos no mejore validación. |
| Deuda, matrícula y beca tienen tasas distintas por nivel. | Podrían añadir señal a los predictores académicos **si existían antes del punto de predicción**. | Que su fecha sea posterior, que dupliquen otra variable o que la mejora desaparezca en validación. |
| `En Curso` representa casi una quinta parte de train. | La clasificación en tres estados necesita evaluación propia. | Que el F1 específico y la matriz de confusión no revelen un problema diferencial; aun así reportar los tres estados. |

Estas hipótesis se formularon mirando train y esta base ya había sido
estudiada en el TP original: son **exploratorias**. El
[EDA extenso](README.md#hipótesis-reformuladas-observación--interpretación--prueba)
expone otras preguntas y criterios que podrían debilitarlas.

## Peldaño 5. ¿Qué variables metemos en el modelo?

**Opción A — el TP con las 27 juntas.** Usamos todos los predictores de la
base en modelos clásicos (como en la resolución original). La pregunta es
cuál clasifica mejor, cuáles son sus errores y cómo cambia la validación
según su complejidad. La importancia por permutación dentro de train
permite examinar dependencia del modelo respecto de cada atributo,
con cautela para las columnas correlacionadas.
[Primera resolución](../../../Resoluci%C3%B3n%20TP%20AA/tp%20completo/TP1_completo.md).

**Opción B — comparar tres momentos sugeridos por los nombres.** Antes de
mirar el target agrupamos 12 variables que podrían conocerse al ingreso,
añadimos cuatro del primer semestre (16) y otras cuatro del segundo (20).
Las **siete restantes** (deuda, matrícula, beca y cuatro familiares) se
analizan por separado porque falta confirmar la fecha. Si se agregan en una
sensibilidad: 19 → 23 → 27. La pregunta es **cuánta capacidad de
clasificación añade cada bloque** usando iguales filas y folds, no elegir
las «mejores doce» a partir de un ranking del EDA.
[Comparación por etapas](../INFORME_ALTERNATIVO.md#5-resultados-de-validación-cruzada).

| Antes del modelo | Durante la validación | Al interpretar |
|---|---|---|
| Definir target, momento hipotético, grupos, métrica y qué contraste responde la pregunta. | Igual partición y folds; ajustes y selección de hiperparámetros dentro de train; comparar con un baseline simple. | Association bivariada ≠ aporte multivariado. Un score mejor con S2 no demuestra que la baja se anticipó: faltan fechas de abandono. |

**Conclusión metodológica:** primero entendemos **todas** las variables;
después reconocemos señales descriptivas y posibles redundancias; recién
entonces comparamos modelos. Ninguna columna se elimina automáticamente
porque su bivariado parezca pequeño o porque otro bivariado sea grande.

## Cómo regenerar y controlar estos resultados

Desde la raíz del repositorio, con la base reconstruida:

```bash
python "AA privada/02_alternativa_temporal/eda/eda_train.py" --input 00_base_de_datos/dataset_TP1_2026.csv
python "AA privada/02_alternativa_temporal/eda/atlas_variables.py" --input 00_base_de_datos/dataset_TP1_2026.csv
```

`atlas_variables.py` verifica el hash de la fuente y de los índices de train
contra `resultados/manifiesto_split.json`. No escribe predicciones por fila,
ni usa el conjunto de test para dibujar relaciones con el target.
