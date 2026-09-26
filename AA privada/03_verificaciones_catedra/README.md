# Comprobaciones a partir de las sugerencias de la cátedra

Esta carpeta **amplía** las dos resoluciones: la tarea principal sigue siendo
clasificar el estado académico con las **27 variables del primer año completo**.
Se utiliza el mismo split multiclase 40.000/10.000 de la alternativa; todos
los cálculos nuevos emplean exclusivamente las 40.000 filas de **train**.
El [script reproducible](experimentos.py), el [CSV por fold](resultados/scores_folds.csv)
y el [protocolo](resultados/protocolo.json) permiten comprobar cada valor. La
base se pasa con `--data 00_base_de_datos/dataset_TP1_2026.csv` desde la raíz del repositorio.

## 1. Curvas: ¿dónde empieza a sufrir la generalización?

Cada punto es la media de **3 folds estratificados**: gris = puntuación en la
parte que entrenó el modelo, rojo = puntuación en el fold que no vio. El
preprocesamiento también se ajusta de nuevo dentro de cada fold. Cambiamos
**un solo hiperparámetro por curva**; el resto permanece fijo. El eje vertical
de las figuras está recortado para leer diferencias pequeñas: revisar los
valores numéricos antes de atribuir importancia visual a una distancia.

| Modelo, 27 variables | Parámetro que varía | Tamaño de train disponible | Configuración fija | Lo que muestra |
|---|---|---:|---|---|
| [Árbol](resultados/curva_arbol.svg) | `max_depth`: 4, 8, 12, sin límite | 40.000 | `min_samples_leaf=50` | De 8 a 12 casi no mejora la validación y aumenta la brecha train–CV. |
| [Random Forest](resultados/curva_randomforest.svg) | `max_depth`: 4, 8, 12, sin límite | 12.000 | 80 árboles, hoja mínima 10, `max_features="sqrt"`, bootstrap | Crece el ajuste en train; CV mejora hasta 12 y luego se aplana. |
| [AdaBoost, primera curva](resultados/curva_adaboost.svg) | `n_estimators`: 20, 60, 120 | 12.000 | árboles base de profundidad 1, `learning_rate=0.1` | Con 20 árboles hay insuficiente ajuste; en esta primera grilla 120 mejora CV. |
| [KNN](resultados/curva_knn.svg) | `n_neighbors`: 3, 11, 31, 101 | 6.000 | escalado dentro del fold | 3 vecinos ajusta demasiado a train; 101 suaviza demasiado. |

En la resolución original ya hay **cinco curvas binarias** en
`Resolución TP AA/informe complejidad y generalización/graficos/`, generadas
por `Resolución TP AA/04_curvas_complejidad_y_generalizacion/04_curvas_complejidad.py`:
Logística (`C`), SVM (`C` y `gamma`) y Árbol (`max_depth` y
`min_samples_leaf`). Allí también se compara train con CV; no hace falta
volver a atribuir esas curvas a este experimento. Aquí agregamos las **cuatro
familias multiclase** que faltaban en la lectura conjunta. Los tamaños 6.000
y 12.000 son pruebas de sensibilidad computacional: sus F1 **no se comparan
directamente** con los resultados publicados sobre 40.000 filas.

| Parámetro (otros fijos) | Train, F1 macro | CV, F1 macro | Brecha train − CV |
|---|---:|---:|---:|
| Árbol, profundidad 4 / 8 / 12 / sin límite | 0,7598 / 0,7808 / 0,7855 / 0,7864 | 0,7594 / 0,7694 / 0,7698 / 0,7696 | 0,0003 / 0,0114 / 0,0157 / 0,0168 |
| RF, profundidad 4 / 8 / 12 / sin límite | 0,7350 / 0,7860 / 0,8027 / 0,8082 | 0,7294 / 0,7685 / 0,7741 / 0,7738 | 0,0056 / 0,0176 / 0,0287 / 0,0344 |
| KNN, vecinos 3 / 11 / 31 / 101 | 0,8229 / 0,7715 / 0,7444 / 0,7067 | 0,7002 / 0,7331 / 0,7242 / 0,7019 | 0,1228 / 0,0383 / 0,0202 / 0,0048 |

**Cómo leerlo:** una mayor flexibilidad *puede* aumentar el sobreajuste, pero
no implica que cada valor adicional empeore CV. Profundidad 4 es más simple
que 12 y, en el RF probado, también rinde peor. El mejor punto se busca
por validación, mirando además la separación train–CV. Para KNN, **menos
vecinos** significa más flexibilidad. Para AdaBoost, más iteraciones no
equivalen automáticamente a sobreajustar: la primera curva llegaba sólo a
120 árboles y usaba profundidad 1; el TP final usa 400 y profundidad 2.

### Extensión de los bordes de K-NN y AdaBoost

Se agregó el [script de sensibilidad](extension_bordes.py), que usa el mismo
split 80/20 y tres folds, exclusivamente en train. Sus salidas son
[K-NN por fold](resultados/extension_knn_cv.csv),
[AdaBoost por fold en 40.000 filas](resultados/extension_ada_cv.csv),
[curva AdaBoost en 12.000](resultados/extension_ada_curva.csv) y el
[protocolo](resultados/extension_bordes_protocolo.json). Mantuvimos fijo el
peso uniforme en K-NN y, en AdaBoost, la profundidad 2 y la tasa 0,1.

| Modelo, 40.000 filas | Valores adicionales frente al borde original | F1 macro CV |
|---|---|---:|
| K-NN | 25 / 35 / 51 vecinos | **0,7487** / 0,7475 / 0,7453 |
| AdaBoost | 200 / 400 / 600 árboles | 0,7623 / 0,7715 / **0,7747** |

La nueva [Figura 5 del informe](../../Resolución%20TP%20AA/tp%20completo/graficos/08_complejidad_multiclase.svg)
usa **profundidad 2 y tasa 0,1** para AdaBoost en 12.000 filas; recorre
100, 200, 400 y 600 árboles. Su F1 de validación es 0,7656 / 0,7716 /
0,7747 / 0,7771. Es una curva de diagnóstico con otra muestra y no
reemplaza la comparación de 40.000 filas ni demuestra sobreajuste: sigue
mejorando. K-NN no mejoró por encima de 25 en la ampliación; AdaBoost
mejoró hasta el nuevo borde. Son observaciones de CV posteriores a la
búsqueda original y no generan una nueva evaluación de test.

## 2. ¿Y si ampliamos la búsqueda a dos hiperparámetros?

Después de mirar cada curva por separado se ensayaron **12 combinaciones**
del Árbol multiclase, todas con las 27 columnas y las 40.000 filas de train:
profundidad `[6, 8, 12, sin límite]` por hoja mínima `[10, 50, 100]`.
Son **36 ajustes**, uno por combinación y fold. La tabla resume tres
contrastes; los doce y sus scores en cada fold están en
[`scores_folds.csv`](resultados/scores_folds.csv), sección `busqueda_2_parametros`.

| Profundidad / mínimo por hoja | F1 macro train | F1 macro CV | Interpretación |
|---|---:|---:|---|
| 12 / 50 | 0,7855 | **0,7698** | Mayor media CV de esta grilla; coincide con la elección del TP original. |
| 8 / 50 | 0,7808 | 0,7694 | Rendimiento CV casi igual, con árbol menos profundo. |
| Sin límite / 10 | 0,8250 | **0,7446** | Ajusta mucho mejor a train y generaliza peor dentro de estos folds. |

**No** es un óptimo universal. La diferencia CV entre 12/50 y 8/50 es
aproximadamente **0,0004** y elegir el máximo entre doce configuraciones
introduce optimismo de selección. Esta grilla comprueba qué pasa al cambiar
dos parámetros **en conjunto**; las curvas anteriores muestran el efecto
de cambiar uno por vez.

## 3. ¿Tres o cinco folds? ¿Y qué aporta el Dummy?

**Mismos modelos y parámetros fijos; misma reserva de test.** Las divisiones
de 3 y 5 folds son distintas: una pequeña diferencia de media no demuestra
que un número de folds sea «mejor». También cambia la fracción disponible
para ajustar dentro de cada fold (2/3 frente a 4/5).

| Tarea y modelo | Filas train disponibles | CV 3 folds | CV 5 folds | Métrica |
|---|---:|---:|---:|---|
| Binaria, Dummy clase frecuente | 40.000 | 0,5000 | 0,5000 | ROC-AUC |
| Binaria, Logística `C=0.1` | 40.000 | 0,9503 | 0,9505 | ROC-AUC |
| Tres clases, Dummy clase frecuente | 40.000 | 0,2144 | 0,2144 | F1 macro |
| Tres clases, Árbol profundidad 12 / hoja 50 | 40.000 | 0,7698 | 0,7691 | F1 macro |
| Tres clases, RF de prueba profundidad 12 / hoja 10 / 80 árboles | 12.000 | 0,7741 | 0,7726 | F1 macro |

El Dummy aprende la **clase más frecuente** del fold; no aprovecha los
predictores y establece una referencia mínima. Superarlo no prueba que un
modelo sea utilizable; hay que mirar F1 por clase, errores y contexto. Ya
había un Dummy en el **test de la alternativa**; aquí se añade su referencia
en CV para compararlo con los otros modelos **en el mismo protocolo**.

## 4. ¿Qué pasa con subconjuntos de datos?

En el [análisis por momentos](../02_alternativa_temporal/INFORME_ALTERNATIVO.md)
se prueban subconjuntos de **variables** (12, 16, 20 y, con siete campos sin
fecha acreditada, hasta 27). Aquí probamos además 10.000, 20.000 y 40.000
filas **del train**, con las 27 columnas, el mismo Árbol (profundidad 12,
hoja 50) y tres folds. Cada tamaño se extrae de las mismas 40.000 filas,
estratificado por clase; no es una cohorte temporal nueva.

| Filas de train disponibles | F1 macro CV | Mínimo–máximo entre folds |
|---:|---:|---:|
| 10.000 | 0,7729 | 0,7620–0,7795 |
| 20.000 | 0,7672 | 0,7558–0,7745 |
| 40.000 | 0,7698 | 0,7641–0,7733 |

**Pregunta:** ¿con más filas mejora necesariamente? En estas tres muestras,
no hay incremento monótono; las fluctuaciones son compatibles con cambios
en las muestras/folds. Tampoco se deduce que 10.000 sean mejores: cada fold
valida personas distintas y en esta base hay registros sintéticamente
aumentados y posibles coincidencias. Para una curva de aprendizaje formal
convendría fijar exactamente los mismos casos de evaluación y variar solo
el conjunto de entrenamiento, con repeticiones.

## 5. OvO frente a OvR: ¿qué habría que contar en el informe?

En la clase de la cátedra, OvO y OvR son **estrategias para entrenar modelos
binarios sobre varias clases**. Si se usaran esas envolturas, OvR ajustaría
tres modelos (cada clase frente al resto) y OvO tres modelos por parejas.
En este TP K-NN, Árbol, Random Forest y AdaBoost predicen tres clases sin
esas envolturas: K-NN vota entre tres etiquetas; Árbol y bosque consideran
las tres proporciones en Gini; AdaBoost combina árboles multiclase. La
consigna pide aclararlo «en los casos que corresponda», sin exigir entrenar
ambas envolturas.

**En los resultados de esta carpeta** OvO y OvR son además nombres de dos
formas de calcular ROC-AUC multiclase a partir de las mismas probabilidades
del mismo modelo; no son dos bosques entrenados de maneras diferentes.
En el código original se informa **OvR** (*one versus rest*):
para cada clase se la compara con la unión de las otras dos y se promedian
las tres AUC. **OvO** (*one versus one*) compara pares de clases y promedia
las comparaciones. «OVC» en el apunte parece aludir a OvR; el nombre del
parámetro usado en los scripts es `multi_class="ovr"`.

| Modelo fijo | Filas train | Folds | AUC macro OvR | AUC macro OvO | F1 macro CV |
|---|---:|---:|---:|---:|---:|
| Árbol, profundidad 12 / hoja 50 | 40.000 | 3 | 0,9159 | 0,9071 | 0,7698 |
| Árbol, profundidad 12 / hoja 50 | 40.000 | 5 | 0,9177 | 0,9089 | 0,7691 |
| RF de prueba, 80 árboles / hoja 10 | 12.000 | 3 | 0,9252 | 0,9161 | 0,7741 |
| RF de prueba, 80 árboles / hoja 10 | 12.000 | 5 | 0,9254 | 0,9164 | 0,7726 |

Ambas AUC indican capacidad para **ordenar probabilidades**, no el acierto
de una etiqueta concreta; por eso también se informa F1 macro y la matriz
de confusión. El hecho de que OvR sea mayor aquí no significa que un
algoritmo distinto ganara: **solo cambió la forma de resumir el score**.
El promedio OvR puede verse afectado por la composición de la clase «resto»;
OvO promedia pares. Véase la [documentación de `roc_auc_score` de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

## 6. ¿Qué hace realmente Random Forest en este TP?

El [código original](../../Resolución%20TP%20AA/07_bagging_y_boosting/07_ensembles.py)
escoge en su búsqueda `n_estimators=500`, `max_depth=None`,
`min_samples_leaf=1`, `max_features="sqrt"`. El bosque construye múltiples
**árboles de decisión**, cada uno sobre una muestra de filas obtenida con
reemplazo (`bootstrap=True`, valor por defecto) y, en cada división, considera
un subconjunto aleatorio de columnas (`sqrt` del total disponible tras el
preprocesamiento). Combina los árboles promediando sus probabilidades por
clase; predice la clase con mayor promedio. Sus árboles base buscan divisiones
que reduzcan impureza Gini de **las tres clases** por defecto. Las muestras
de filas y columnas hacen menos parecidos a los árboles: promediar árboles
diversos reduce la variación respecto de usar uno solo. **No** usa una regresión logística ni
un modelo SVM dentro de cada árbol. Documentación:
[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

En esta carpeta el **bosque de prueba** usa 80 árboles y hoja mínima 10 para
hacer visible una curva a coste moderado; por ello sus F1 no reemplazan el
**0,7808 de F1 macro en test del bosque original**. La
[permutación original](../COMPARACION_RESULTADOS.md#2-primera-resolución-qué-algoritmos-rindieron-mejor-con-las-27)
indica que el bosque depende en especial de aprobadas del segundo semestre,
matrícula al día, nota del segundo semestre y aprobadas del primero. Esa
importancia describe el modelo y **no** establece una causa del abandono.

## 7. Cómo elegir la próxima grilla sin perder el test

1. Plantear **1 o 2 parámetros con significado** y un rango corto a partir
   de las curvas; el Árbol anterior ya prueba dos simultáneamente. La
   ampliación de K-NN y AdaBoost anterior prueba valores más allá de los
   bordes originales: en AdaBoost 600 todavía es el extremo mejor.
2. Si se amplía el espacio de búsqueda, comparar **por CV solo en train**;
   registrar la métrica, dispersión entre folds y brecha train–CV. Un
   `GridSearchCV` grande aumenta coste y puede favorecer al ganador por
   azar de esa misma CV; una CV externa sería mejor para estimar el resultado
   de todo el procedimiento de selección.
3. Mantener el test reservado. Como el test de **esta base ya se examinó** en
   ambas resoluciones, una mejora futura aquí debe describirse como
   exploratoria; una estimación nueva requeriría una cohorte externa.

**Respuesta breve al apunte:** sí, hay curvas para las familias binaria y
multiclase, se ensayan uno o dos hiperparámetros con una razón para cada
rango, hay Dummy, 3 frente a 5 folds, subconjuntos de columnas y de filas,
y una comparación explícita de OvR/OvO. Ninguno de estos experimentos
adicionales altera la primera resolución ni sustituye el objetivo con las 27.
