# Cinco lecturas del EDA de train

**Alcance:** son teorías de trabajo *exploratorias*, formuladas al analizar una
base que ya se empleó en el TP original. El EDA condicionado por el target de
esta carpeta usa únicamente las 40.000 filas de train, pero no podemos llamar
al test una cohorte completamente nueva. Una asociación descriptiva o un
predictor importante no determina una causa ni una acción efectiva.

Las siete hipótesis formales H1–H7, con contrastes y criterios que las
debilitarían, están en `eda/README.md`. Estos cinco ejes desarrollan su
interpretación sin crear otra numeración de hipótesis.

## Perfil al ingreso: relación con H4 del EDA

**Observación:** la mediana de edad es 22 años en Desertor y 19 en En Curso y
Graduado. La mediana del puntaje de ingreso es 61, 61 y 63. Las tasas de
deserción cambian entre carreras: Enfermería 13,63 % (n=6.297) y Animación y
Diseño Multimedia 71,30 % (n=1.474), calculadas en train. Algunas categorías
son diminutas; sus tasas no merecen la misma confianza.

**Teoría plausible:** condiciones de ingreso y composición de carreras se
relacionan con la trayectoria, pero pueden ser marcadores de procesos no
medidos. Edad, carrera y modalidad no son causas identificadas.

**Mirar precisamente:** denominador y tasa por categoría, incertidumbre de
categorías pequeñas, diferencias *dentro* de carreras, prevalencia de las
tres clases y score de CV de E0 frente a un Dummy. Si E0 es cercano al
baseline, la hipótesis de utilidad temprana no tiene apoyo práctico.

## Primer semestre: relación con H1 del EDA

**Observación:** aprobadas S1 es cero en 58,15 % de Desertor (7.689/13.223),
4,20 % de En Curso (328/7.811) y 1,47 % de Graduado (279/18.966). Entre
valores positivos, las medianas de aprobadas son 4, 5 y 6, respectivamente.

**Teoría plausible:** una gran parte de la separación aparece como ausencia
de aprobaciones; otra parte aparece entre quienes sí aprueban materias. Cero
aprobadas no distingue por sí solo falta de inscripción, evaluaciones no
rendidas, exámenes desaprobados o abandono previo a la medición.

**Mirar precisamente:** porcentaje cero con denominador por clase,
matriculación y evaluaciones condicionadas al cero, mediana/IQR solo entre
positivos, tasas de deserción en grupos de actividad y diferencia E1−E0
en los mismos folds. La importancia por permutación de la variable entera
no separaría el efecto de cero del nivel positivo.

## Segundo semestre: relación con H2 y H7 del EDA

**Observación:** aprobadas S2 es cero en 67,90 % de Desertor (8.978/13.223),
4,62 % de En Curso (361/7.811) y 1,50 % de Graduado (284/18.966). Entre
positivos las medianas son 3, 4 y 6. Entre las personas inscritas en ambos
semestres y con aprobación S1 positiva, el grupo que pasa a cero en S2 tiene
n=1.545 y 92,43 % etiquetado Desertor; el grupo 0→positivo n=217 también
tiene 64,06 % Desertor. La transición no es una regla determinista.

**Teoría plausible:** el registro del segundo semestre capta una trayectoria
ya avanzada o incluso un abandono ocurrido antes de ese registro. Una
mejora predictiva E2−E1 no demuestra capacidad de *alerta previa*.

**Mirar precisamente:** transiciones 0/positivo S1→S2 con cuatro celdas y
denominadores, notas entre positivos, redundancia S1/S2, diferencia CV
E2−E1 en folds pareados y fecha del abandono relativa a S2. Esta última
fecha **no está en la base**, por lo que la anticipación no se puede probar.

## Variables de fecha incierta: relación con H3 y H5 del EDA

**Observación:** matrícula no al día tiene tasa Desertor 93,79 % (n=4.332),
deuda Sí 76,05 % (n=2.877), beca Sí 7,21 % (n=9.930), todo en train. El
diccionario no fecha estos campos; tampoco fecha cuatro indicadores
parentales. Las parejas demanda cognitiva/ingresos del mismo progenitor
tienen correspondencia categórica exacta en esta base.

**Teoría plausible:** matrícula y beca son asociaciones fuertes que podrían
representar recursos, reglas administrativas o incluso consecuencias de la
trayectoria. Las variables parentales duplicadas no ofrecen dos piezas de
evidencia independientes.

**Mirar precisamente:** fecha de captura y definición administrativa;
denominadores y tasas dentro de carrera/edad; diferencia E0+?−E0,
E1+?−E1 y E2+?−E2 en CV; redundancia entre columnas. Hasta conocer la
fecha, **no** llamar a E0+? un modelo de ingreso.

## «En Curso»: relación con H6 del EDA

**Observación:** En Curso reúne 7.811/40.000 (19,53 %) en train, y sus
medianas de aprobadas positivas quedan entre Desertor y Graduado. Es una
etiqueta de estado, no un desenlace final equivalente a las otras dos.

**Teoría plausible:** parte de su solapamiento refleja la definición del
target y el momento de seguimiento, además de dificultades del algoritmo.

**Mirar precisamente:** matriz de confusión de tres clases, precision,
recall y F1 específicos de En Curso; macro F1 para evitar que Graduado
domine; horizonte de seguimiento y reglas de etiquetado, que este CSV no
documenta. Un buen binario puede ocultar la mala identificación de En Curso.

## Criterios de lectura

Las diferencias E1−E0 y E2−E1 se calculan con **los mismos individuos,
los mismos folds y la misma familia de modelo por tarea**. En la búsqueda
principal se vuelve a seleccionar la complejidad óptima dentro de cada
etapa; la validación con parámetros fijos cuantifica la sensibilidad a esa
decisión. Los resultados de test son descriptivos y se señalan aparte.

Una hipótesis descriptiva sobre ceros no queda comprobada solo por una
mejora del score al sumar cuatro columnas. Haría falta aislar las partes
«cero» y «nivel positivo» dentro de train. Una hipótesis causal requeriría
un diseño y mediciones adicionales.
