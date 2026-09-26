# EDA comentado: entender las 27 variables, una por una

Esta es una guía de lectura de los [27 gráficos del atlas](atlas/README.md), no una selección previa de variables «buenas». La pregunta del TP es clasificar el estado académico a partir de características de ingreso y del primer año. Aquí describimos **todas** las columnas antes de elegir cómo comparar modelos. El [mapa de asociaciones](MAPA_ASOCIACIONES_27.md) las reúne por tipo y explica con qué evidencia se relaciona cada una con el target.

## 1. Cómo llegamos a las respuestas

Hay **11 variables numéricas y 16 categóricas**; `target` es una columna aparte y no cuenta entre las 27. La base tiene 50.000 filas; este EDA usa **40.000 de entrenamiento**, separadas con `test_size=0.20`, `stratify=target` y semilla 42. Las otras 10.000 se apartaron antes de explorar. El target tiene tres categorías: **Desertor** 13.223 (33,1 %), **En Curso** 7.811 (19,5 %) y **Graduado** 18.966 (47,4 %). Estas etiquetas describen la base: no conocemos la fecha individual del abandono. [Gráfico de las clases](resultados/01_clases_train.png).

**Primera pregunta, sin target:** ¿qué representa la variable, cuántas personas hay por valor, hay ceros, valores raros o grupos pequeños? En numéricas miramos distribución, rango, mediana y cuartiles; en categóricas contamos niveles. **Segunda pregunta, con target:** ¿cómo cambian los valores entre los tres estados? Para numéricas comparamos distribuciones por clase; para categóricas miramos la proporción de cada estado **dentro de cada nivel**, siempre con su tamaño `n`. Una diferencia puede venir de una tercera variable, de cómo se construyó esta base o de cuándo se midió la columna. Ni el gráfico ni una asociación prueban causalidad o utilidad adicional dentro de un modelo.

En los gráficos numéricos, la izquierda contiene la distribución total y la derecha las medianas y cuartiles por clase; para los **ocho indicadores académicos**, la derecha muestra sólo valores **mayores que cero**. Se informa por separado cuánto pesa el cero. En los categóricos, la izquierda muestra tamaños de niveles y la derecha `% Desertor` dentro del nivel. La línea de referencia es 33,1 %; el rojo subraya tasas altas en grupos de al menos 100. **Gris no significa ausencia de relación**; también puede significar tasa baja o grupo pequeño. Las tablas [numéricas](resultados/03_numericas_por_clase_train.csv) y [categóricas](resultados/04_categoricas_por_clase_train.csv) dan todos los cuartiles, tamaños y porcentajes de los tres estados.

Un ejemplo para no invertir denominadores: de los 8.296 estudiantes con **cero aprobadas en S1**, 92,7 % son Desertor: `P(Desertor | aprobadas=0)`. A la inversa, **58,1 % de los 13.223 Desertor** tienen cero aprobadas en S1: `P(aprobadas=0 | Desertor)`. Ambas cifras son correctas y contestan preguntas diferentes. Además, el cero puede significar no haber cursado o rendido, y no conocemos aquí su definición institucional.

## 2. Antecedentes e ingreso: tres numéricas

Estas tres columnas parecen medirse al ingreso por su nombre; la disponibilidad operativa debe confirmarse. Las medianas iguales no excluyen diferencias en las colas ni en las combinaciones con otras variables.

### 01. Calificación promedio de estudios previos — numérica

**Qué preguntamos:** ¿qué trayectoria académica previa aparece y separa los estados? **Sin target:** valores 48–95, mediana 67, mitad central 62–70. **Con target:** medianas Desertor 67 (cuartiles 62–68), En Curso 65 (62–68), Graduado 67 (64–70). Hay mucho solapamiento: la mediana por sí sola no permite identificar a un desertor. **Mirar después:** comparar distribuciones dentro de cada modalidad de admisión; ¿la diferencia de cuartiles se mantiene? ![Gráfico descriptivo 01](atlas/figuras/01.svg).

### 02. Puntaje en examen de ingreso — numérica

**Qué preguntamos:** ¿se distinguen los estados por puntaje de entrada? **Sin target:** valores 48–95, mediana 62. **Con target:** medianas Desertor 61, En Curso 61 y Graduado 63; los cuartiles son 58–65, 59–65 y 60–67, respectivamente. El desplazamiento es pequeño frente al solapamiento. **Mirar después:** si el puntaje aporta información por encima de la calificación previa, usando los mismos estudiantes y validación; no declarar «inútil» a partir de una mediana. ![Gráfico descriptivo 02](atlas/figuras/02.svg).

### 03. Edad al inscribirse — numérica

**Qué preguntamos:** ¿qué edades hay y en qué estados se concentran? **Sin target:** 17–70 años, mediana 19. **Con target:** Desertor tiene mediana 22 años (cuartiles 19–29); En Curso y Graduado, 19 (18–21 y 18–20). La cola de edades mayores pesa más en Desertor. **Mirar después:** tasa de deserción por rangos de edad con tamaño visible, y repetir dentro de carrera, modalidad y turno para distinguir composiciones; ninguna edad «causa» abandono por este gráfico. ![Gráfico descriptivo 03](atlas/figuras/03.svg).

## 3. Primer semestre: cuatro numéricas

Estos indicadores se **atribuyen al cierre de S1 por su nombre**, no a la inscripción. Separar **cero** de **magnitud entre positivos** es crucial. Comparar grupos por estado no indica en qué fecha ocurrió la deserción.

### 04. Materias inscriptas en S1

**Qué preguntamos:** ¿había carga académica registrada? **Sin target:** 0–26, mediana 6; el 3,5 % tiene cero. **Con target:** la mediana entre quienes tienen al menos una materia es **6 en los tres estados**; el cero aparece en 7,8 % de Desertor frente a 1,1 % En Curso y 1,3 % Graduado. La diferencia visible se concentra en la ausencia de inscripción, más que en la mediana de carga positiva. **Mirar después:** si se trata de matrícula tardía, errores o un estado posterior; las aprobadas pueden contener más señal que la carga. ![Gráfico descriptivo 04](atlas/figuras/04.svg).

### 05. Cantidad de evaluaciones en S1

**Qué preguntamos:** ¿hay actividad evaluativa y cuánta? **Sin target:** 0–45, mediana de todos 7, 10,2 % con cero. **Con target:** cero en 27,9 % Desertor, 1,5 % En Curso y 1,4 % Graduado; **entre positivos**, medianas 8, 9 y 7. Más evaluaciones no implica necesariamente menor abandono: pueden variar las exigencias de cada carrera o la cantidad de intentos. **Mirar después:** relacionar evaluaciones con materias inscriptas y aprobadas dentro de cada carrera. ![Gráfico descriptivo 05](atlas/figuras/05.svg).

### 06. Materias aprobadas en S1

**Qué preguntamos:** ¿cuánto progreso hay y qué significa no registrar aprobaciones? **Sin target:** 0–26, mediana de todos 5, 20,7 % con cero. **Con target:** cero en 58,1 % de Desertor, 4,2 % En Curso y 1,5 % Graduado; **entre positivos**, mediana 4, 5 y 6. Son dos diferencias: presencia de alguna aprobación y número de aprobadas una vez que existe progreso. **Mirar después:** contrastar `0`, `1–3` y `4+` dentro de carrera y turno, y confirmar el significado del cero. Hay registros con aprobadas > inscriptas. ![Gráfico descriptivo 06](atlas/figuras/06.svg).

### 07. Nota promedio en S1

**Qué preguntamos:** ¿el cero representa rendimiento o ausencia de nota? **Sin target:** 0–90, mediana de todos 61; cero en 20,7 %. **Con target:** cero en 58,1 % Desertor, 4,2 % En Curso y 1,5 % Graduado; **entre notas positivas**, medianas 59, 60 y 66. El patrón de ceros casi coincide con aprobadas=0: podrían repetir una misma condición de actividad. **Mirar después:** tablas cruzadas nota=0 × aprobadas=0 y desempeño entre quienes efectivamente tienen nota; no presentar una nota cero automáticamente como examen desaprobado. ![Gráfico descriptivo 07](atlas/figuras/07.svg).

## 4. Segundo semestre: cuatro numéricas

Repetimos exactamente las mismas preguntas. La mayor asociación puede deberse a que estas mediciones están **más cerca del estado final**; sin fecha de abandono, eso no prueba detección temprana.

### 08. Materias inscriptas en S2

**Qué preguntamos:** ¿sigue habiendo inscripción? **Sin target:** 0–21, mediana 6, 3,5 % con cero. **Con target:** cero en 7,9 % Desertor, 1,1 % En Curso y 1,3 % Graduado; mediana entre positivos 6 en las tres clases. El patrón es parecido al de S1. **Mirar después:** transición S1→S2 por persona: ¿quiénes dejaron de registrar materias?, ¿cómo se define la inscripción en cada semestre? ![Gráfico descriptivo 08](atlas/figuras/08.svg).

### 09. Cantidad de evaluaciones en S2

**Qué preguntamos:** ¿se registra actividad evaluativa en el segundo semestre? **Sin target:** 0–33, mediana de todos 8, 11,3 % con cero. **Con target:** cero en 31,3 % Desertor, 1,6 % En Curso y 1,4 % Graduado; entre positivos, medianas 8, 9 y 7. Tener más evaluaciones entre positivos no debe traducirse mecánicamente como éxito. **Mirar después:** contrastar ceros de S1 y S2 y la carga por carrera; considerar intentos de examen y distintas exigencias. ![Gráfico descriptivo 09](atlas/figuras/09.svg).

### 10. Materias aprobadas en S2

**Qué preguntamos:** ¿aparece progreso o desaparece? **Sin target:** 0–19, mediana de todos 5, 24,1 % con cero. **Con target:** cero en 67,9 % Desertor, 4,6 % En Curso y 1,5 % Graduado; entre positivos, medianas 3, 4 y 6. Entre las personas con cero en S2, 93,3 % son Desertor; ese porcentaje describe la base y no una regla validada para actuar. **Mirar después:** las trayectorias `0→0`, `0→positivo`, `positivo→0`, `positivo→positivo` en [tabla de transiciones](resultados/08_trayectorias_academicas_train.csv), con denominadores. ![Gráfico descriptivo 10](atlas/figuras/10.svg).

### 11. Nota promedio en S2

**Qué preguntamos:** ¿separa estados la nota una vez que existe? **Sin target:** 0–89, mediana de todos 61; 24,1 % tiene cero. **Con target:** cero en 67,9 % Desertor, 4,6 % En Curso y 1,5 % Graduado; mediana **positiva** 58, 60 y 66. De nuevo se mezclan ausencia de nota y nivel de nota; la señal puede solaparse con aprobadas de S2. **Mirar después:** comparar nota y aprobadas de S2 entre sí y hacer una comparación de modelos por bloques, sin separar información de una misma persona entre train y validación. ![Gráfico descriptivo 11](atlas/figuras/11.svg).

## 5. Contexto y antecedentes: nueve categóricas

En cada caso la tasa es `n Desertor en ese nivel / n total de ese nivel`. Un nivel infrecuente puede tener una tasa llamativa por pocas observaciones; consultar la [tabla completa con intervalos](resultados/04_categoricas_por_clase_train.csv). Varias parecen conocidas al ingreso; la fecha exacta debe verificarse para uso temporal.

### 12. Estudios máximos antes de inscribirse

**Qué preguntamos:** ¿qué antecedentes educativos predominan? **Sin target:** Educación Secundaria 35.109 de 40.000; Superior Pregrado 2.665; Educación Básica 2.186; Posgrado 33 y Otros/No completado 7. **Con target:** Desertor 29,5 % dentro de Secundaria, 43,0 % en Pregrado y 77,4 % en Básica. Los grupos de 33 y 7 no permiten una conclusión sólida por nivel. **Mirar después:** edad y modalidad dentro de cada antecedente; no imponer una relación monótona sólo por el nombre de las categorías. ![Gráfico descriptivo 12](atlas/figuras/12.svg).

### 13. Estado civil

**Qué preguntamos:** ¿qué estados existen y cuántas personas representan? **Sin target:** Soltero 36.712, Casado 2.730, Divorciado 471; Unión de hecho 58, Separado legalmente 18, Viudo 11. **Con target:** Desertor 30,9 % entre solteros, 56,5 % casados y 61,8 % divorciados. Los demás niveles son escasos. **Mirar después:** comparar dentro de edades y modalidad de ingreso; el diccionario ubica esta medición al inscribirse, pero la diferencia no explica su causa. ![Gráfico descriptivo 13](atlas/figuras/13.svg).

### 14. Modo de aplicación o admisión

**Qué preguntamos:** ¿cómo ingresó cada estudiante? **Sin target:** Acceso General 27.723, Normativa Específica 11.295, Acceso para Titulados 799, Contingente Especial Regional 183. **Con target:** Desertor 22,7 %, 56,0 %, 72,1 % y 13,7 % dentro de esos niveles, respectivamente. Son perfiles muy distintos en esta base. **Mirar después:** distribución de carrera, edad y estudios previos en cada modo; contrastar dentro de grupos comparables. ![Gráfico descriptivo 14](atlas/figuras/14.svg).

### 15. Sexo registrado

**Qué preguntamos:** ¿cuáles son los tamaños de los niveles codificados? **Sin target:** Femenino 27.392 y Masculino 12.608; el archivo sólo presenta esas categorías. **Con target:** Desertor 23,7 % en Femenino y 53,4 % en Masculino. Esta diferencia descriptiva no justifica atribuir el resultado al sexo: las carreras, turnos y modalidades podrían estar repartidos de forma distinta. **Mirar después:** tasas por carrera y auditoría de rendimiento del modelo por grupo; no inferir identidades o mecanismos que la base no documenta. ![Gráfico descriptivo 15](atlas/figuras/15.svg).

### 16. Desplazado

**Qué preguntamos:** ¿cambia la distribución para quienes debieron mudarse de localidad para cursar, según el diccionario? **Sin target:** Sí 22.891, No 17.109. **Con target:** Desertor 27,3 % con Sí y 40,8 % con No. Ser desplazado no se puede interpretar aquí como «protector»: falta establecer la fecha y las diferencias de composición. **Mirar después:** origen del registro, carrera, edad y modalidad en los dos niveles. ![Gráfico descriptivo 16](atlas/figuras/16.svg).

### 17. Carrera

**Qué preguntamos:** ¿todas las carreras tienen igual tamaño y mezcla de estados? **Sin target:** 15 carreras; Enfermería 6.297, Gestión 6.192, Servicio Social 5.434; Tecnología en Producción de Biocombustibles, apenas 38. **Con target:** Desertor 13,6 % en Enfermería, 42,0 % en Gestión, 59,4 % en Ingeniería Informática (n=1.588), 71,3 % en Animación y Diseño Multimedia (n=1.474). En Ingeniería Informática también hay 35,6 % En Curso y 5,0 % Graduado: reducir todo a Desertor vs resto oculta esta diferencia. **Mirar después:** analizar las tres clases y separar efectos de carrera de composición por modo y edad. Evitar conclusiones sobre los 38 casos de Biocombustibles. ![Gráfico descriptivo 17](atlas/figuras/17.svg).

### 18. Turno diurno/nocturno

**Qué preguntamos:** ¿cuánto pesa cada turno? **Sin target:** Diurna 36.633, Nocturna 3.367. **Con target:** Desertor 31,2 % en Diurna y 52,9 % en Nocturna. **Mirar después:** edad y carrera dentro de ambos turnos y cuándo se registró la modalidad; no atribuir la diferencia al horario sin comparaciones adicionales. ![Gráfico descriptivo 18](atlas/figuras/18.svg).

### 19. Necesidades educativas especiales

**Qué preguntamos:** ¿hay suficientes observaciones para comparar? **Sin target:** Sí 158, No 39.842. **Con target:** Desertor 33,5 % en Sí y 33,1 % en No. Esa similitud no demuestra que las necesidades no importen: el grupo Sí es pequeño y las condiciones de registro no están documentadas. **Mirar después:** intervalos, cobertura y desempeño del clasificador por grupo si se usa esta variable; evitar inferencias individuales. ![Gráfico descriptivo 19](atlas/figuras/19.svg).

### 20. Estudiante internacional

**Qué preguntamos:** ¿es frecuente esta categoría? **Sin target:** Sí 257 y No 39.743. **Con target:** Desertor 32,3 % frente a 33,1 %. La diferencia es muy pequeña en una categoría minoritaria; no constituye evidencia de equivalencia. **Mirar después:** intervalos, procedencia de la codificación y representación en train/validación. ![Gráfico descriptivo 20](atlas/figuras/20.svg).

## 6. Situación administrativa: tres categóricas con fecha desconocida

Se ven asociaciones grandes, pero **no sabemos si la variable fue medida al ingreso, durante el año o después**. No situarla en la etapa «temprana» hasta verificar la captura y la definición del target; puede existir fuga de información temporal.

### 21. Tiene deuda — Sí/No

**Qué preguntamos:** ¿qué proporción debe y cómo se distribuye el target? **Sin target:** Sí 2.877 (7,2 %), No 37.123. **Con target:** **76,1 %** de quienes tienen deuda son Desertor frente a **29,7 %** de quienes no. Es `P(Desertor | deuda=Sí)`, no el porcentaje de desertores endeudados. **Mirar después:** fecha en que se generó la deuda, momento de extracción y posible solapamiento con pago de matrícula; no decir que la deuda causó la deserción. ![Gráfico descriptivo 21](atlas/figuras/21.svg).

### 22. Pago de matrícula al día — Sí/No

**Qué preguntamos:** ¿qué registra exactamente «al día» y cuándo? **Sin target:** Sí 35.668, No 4.332. **Con target:** Desertor **25,7 %** entre quienes figuran al día y **93,8 %** entre quienes no. La separación muy grande exige descartar que el estado de pago haya sido actualizado después del abandono. **Mirar después:** fecha de corte, cruce con deuda y definición administrativa, antes de proponer su uso para anticipar deserción. ![Gráfico descriptivo 22](atlas/figuras/22.svg).

### 23. Poseedor de beca — Sí/No

**Qué preguntamos:** ¿qué proporción tiene beca y cuál es su mezcla de estados? **Sin target:** Sí 9.930 y No 30.070. **Con target:** Desertor **7,2 %** entre becados y **41,6 %** entre no becados. No equivale a medir un efecto causal de conceder becas; selección, requisitos o vigencia de la beca podrían explicar parte del contraste. **Mirar después:** momento de otorgamiento, criterios y carrera; examinar si agrega señal válida una vez conocidas esas fechas. ![Gráfico descriptivo 23](atlas/figuras/23.svg).

## 7. Familia: cuatro categóricas ordinales con fecha desconocida

Los niveles se ordenan, pero **no son cantidades monetarias ni puntajes continuos**. La correspondencia exacta de cada par cognición/ingresos en este train es un hallazgo de **calidad/redundancia de la base**, no una teoría sobre las familias. No podemos asignar estas cuatro columnas al ingreso sin verificar su fecha.

### 24. Demanda cognitiva del padre

**Qué preguntamos:** ¿hay una progresión de tasas de muy baja a muy alta? **Sin target:** Muy baja 15.987, Baja 8.247, Media 8.542, Alta 3.018 y Muy alta 4.206. **Con target:** Desertor 36,9 %, 27,4 %, 32,1 %, 29,1 % y 34,5 % en ese orden. No aparece un descenso o ascenso regular. **Mirar después:** definición original del atributo y comparación fila a fila con ingresos del padre. ![Gráfico descriptivo 24](atlas/figuras/24.svg).

### 25. Demanda cognitiva de la madre

**Qué preguntamos:** ¿son iguales los tamaños y el patrón al del padre? **Sin target:** Muy baja 18.694, Baja 2.605, Media 13.368, Alta 2.443, Muy alta 2.890. **Con target:** Desertor 37,3 %, 30,9 %, 29,4 %, 24,0 % y 32,3 %. El último nivel rompe una lectura monotónica. **Mirar después:** incertidumbre por nivel, definición y cruce con ingresos de la madre; las distribuciones materna y paterna no son iguales. ![Gráfico descriptivo 25](atlas/figuras/25.svg).

### 26. Ingresos del padre

**Qué preguntamos:** ¿aportan algo distinto de la columna 24? **Sin target:** Bajo 15.987, Medio-bajo 8.247, Medio 8.542, Medio-alto 3.018 y Alto 4.206. **Con target:** tasas Desertor 36,9 %, 27,4 %, 32,1 %, 29,1 % y 34,5 %, idénticas por recodificación a las de demanda cognitiva paterna. Cada categoría de una columna determina exactamente la categoría de la otra **en train**. **Mirar después:** origen de ambas mediciones; comparar modelos con ambas o una por par, sin contar una «señal nueva» dos veces. ![Gráfico descriptivo 26](atlas/figuras/26.svg).

### 27. Ingresos de la madre

**Qué preguntamos:** ¿se repite la redundancia en el otro par? **Sin target:** Bajo 18.694, Medio-bajo 2.605, Medio 13.368, Medio-alto 2.443 y Alto 2.890. **Con target:** Desertor 37,3 %, 30,9 %, 29,4 %, 24,0 % y 32,3 %, igual que en demanda cognitiva materna tras cambiar las etiquetas. **Mirar después:** verificar mecanismo de construcción; la igualdad observada de estos pares no autoriza a afirmar que ingresos y cognición son lo mismo fuera de esta base. ![Gráfico descriptivo 27](atlas/figuras/27.svg).

## 8. ¿Qué aprendimos y cómo nace una hipótesis?

| Evidencia descriptiva | Hipótesis que vale la pena comprobar | Prueba concreta y posible refutación |
|---|---|---|
| Los ceros y las aprobadas positivas separan estados en S1 y S2. | Los bloques académicos añaden información por encima del ingreso. | Mismas filas, folds, preprocesamiento y clasificador; comparar ingreso solo, +S1 y +S2. Si la validación no mejora o cambia erráticamente, la hipótesis pierde apoyo. |
| Edad al ingreso difiere por clase; puntajes previos se solapan. | El ingreso podría ofrecer señal limitada antes de cursar. | Evaluar baseline e ingreso con CV; estratificar edad por carrera y modo. Una diferencia bivariada no asegura utilidad multivariada. |
| Deuda, matrícula y beca separan mucho las tasas. | Podrían aportar señal si su fecha precede el instante de predicción. | Primero verificar fechas. Si se registran después o dependen del abandono, no sirven para esa predicción aunque suba el score. |
| Las variables paternas forman un par de recodificaciones; lo mismo las maternas. | Quitar una copia de cada par podría mantener desempeño y simplificar interpretación. | Comparar por validación con ambas y con una por par; investigar por qué se duplicaron. |
| `En Curso` representa 19,5 % y varía entre carreras. | Un modelo binario puede ocultar errores importantes en ese estado. | Evaluar también clasificación de tres clases, F1 por clase y matriz de confusión sobre predicciones. |

**Cómo detectar candidatas:** empezar por cobertura, calidad, diferencias descriptivas con denominadores suficientes y fecha de disponibilidad; proponer hipótesis con posibles explicaciones alternativas. Mirar la [asociación marginal](resultados/05_asociaciones_marginales_train.csv) **dentro de cada tipo**: η² en numéricas y V de Cramér en categóricas tienen definiciones diferentes y no se ordenan en una misma escala. Después comprobar redundancia [entre predictores](resultados/06_redundancia_train.csv). Finalmente probar la **aportación adicional** con validación y un modelo: una variable con gran contraste bivariado puede repetir otra, y una de contraste pequeño puede ayudar al combinarse. EDA no exige empezar por regresión logística ni obliga a excluir columnas.

**Dos decisiones de modelado legítimas:** (A) usar las **27 juntas**, como el TP original, para comparar algoritmos y errores en la tarea planteada; (B) comparar un conjunto *candidato* de **12 de ingreso**, luego **16 con S1** y **20 con S2**, para estudiar qué cambia al incorporar bloques. Las otras **7** (deuda, matrícula, beca y las cuatro familiares) se evalúan aparte por fecha no verificada; también existe el análisis de sensibilidad 19→23→27. Estas 12 no fueron elegidas por tener la asociación más alta: se agruparon por **disponibilidad sugerida por nombres/diccionario**. Antes de afirmar un sistema temprano se necesitan fechas de captura y de abandono. [Diseño y resultados de la comparación](../INFORME_ALTERNATIVO.md).

**Controles pendientes y límites:** no hay `NaN` codificados, pero los ceros académicos necesitan definición. En train hay **213 filas exactamente duplicadas**, **235 patrones de predictores repetidos**, **22 patrones que aparecen con targets distintos** (49 filas) y **34 casos en S1 / 62 en S2 con aprobadas superiores a inscriptas**; ver [chequeos](resultados/07_chequeos_train.csv). La base es sintéticamente aumentada. La exploración y las hipótesis ya se elaboraron sobre datos usados en la resolución original, así que las cifras posteriores de test no son confirmación completamente independiente de todas estas ideas. Una matriz de confusión se calcula **después** de entrenar y predecir, no durante esta descripción.

**Fuente reproducible:** [diccionario de 27 columnas](../../../00_base_de_datos/diccionario_datos_TP1_2026.csv), [manifiesto de fuente y split](resultados/manifiesto_split.json), [script de tablas y controles](eda_train.py), [script de los 27 gráficos](atlas_variables.py), [tablas completas de todos los niveles](resultados/). Los gráficos mantienen gris/rojo y un foco por figura; este texto explica lo que cada uno permite y no permite concluir.
