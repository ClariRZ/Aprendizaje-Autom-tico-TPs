# Decisiones metodológicas — Complejidad y generalización

## 1. Objetivo

El bloque 04 no vuelve a elegir modelos. Su objetivo es **interpretar cómo cambia la generalización cuando varía la complejidad**.

La pregunta es:

> ¿en qué zona el modelo queda corto, en qué zona generaliza bien y en qué zona comienza a ajustar demasiado al entrenamiento?

---

## 2. Test permanece fuera

**Decisión:** las curvas se calculan exclusivamente sobre el 80% reservado para entrenamiento.

Dentro de ese conjunto se usa validación cruzada estratificada de 3 folds.

**Por qué:** el test ya fue utilizado en el bloque 03 para la evaluación final. No debe reutilizarse para elegir nuevas configuraciones.

---

## 3. Mismo esquema de validación que en el bloque 03

**Decisión:** StratifiedKFold de 3 folds, shuffle y semilla 42.

**Por qué:** mantiene la comparación coherente con la selección de hiperparámetros previa y conserva la proporción de Desertor en cada fold.

---

## 4. Una dimensión de complejidad por vez

**Decisión:** al estudiar un hiperparámetro, los demás se fijan en el valor seleccionado en el bloque 03.

Esto permite atribuir el cambio observado a una dimensión concreta de complejidad.

### Logística
Se varía C.

### SVM
- se varía C con gamma = 0,01;
- se varía gamma con C = 0,1.

### Árbol
- se varía max_depth con min_samples_leaf = 50;
- se varía min_samples_leaf con max_depth = 8.

---

## 5. Corrección respecto de la primera versión

La primera versión del script de complejidad analizaba sólo el Árbol y variaba `max_depth` con `min_samples_leaf=1`.

Eso era incoherente con el árbol seleccionado en el bloque 03:

    max_depth = 8
    min_samples_leaf = 50

La versión final corrige esa inconsistencia.

---

## 6. Cómo se interpreta una curva

### Underfitting
Train y validación presentan rendimiento relativamente bajo.

### Zona útil
La validación alcanza valores altos y la brecha train-validación se mantiene pequeña.

### Overfitting
El entrenamiento sigue mejorando mientras la validación se estanca o empeora, aumentando la brecha.

Esta definición se aplica con mayor claridad al árbol. En Logística y SVM, la relación entre el hiperparámetro y ROC-AUC no tiene por qué ser perfectamente monótona porque el algoritmo no optimiza directamente ROC-AUC.

---

## 7. Regresión Logística

Valores evaluados:

    C = [0,01; 0,1; 1; 10]

La validación fue:

- C=0,01 → 0,95036
- C=0,1 → 0,95069
- C=1 → 0,95063
- C=10 → 0,95062

**Decisión interpretativa:** no describir diferencias de diezmilésimas como cambios sustantivos.

La curva muestra una meseta amplia. C=0,1 —seleccionado previamente— se encuentra dentro de esa zona estable.

---

## 8. SVM — C

Con gamma=0,01:

- C=0,1 → validación 0,94617; gap 0,00071
- C=1 → validación 0,93736; gap 0,00326
- C=10 → validación 0,93266; gap 0,01049

**Interpretación:** aumentar C no mejora la discriminación medida por ROC-AUC. La validación empeora y aumenta la brecha train-validación.

No se describe como un patrón perfectamente clásico de overfitting porque el ROC-AUC de train tampoco aumenta de manera monotónica; SVC optimiza su función de margen, no ROC-AUC directamente.

---

## 9. SVM — gamma

Con C=0,1:

- gamma=scale → validación 0,93733
- gamma=0,01 → validación 0,94617
- gamma=0,1 → validación 0,93477

Gamma=0,01 presenta la mejor validación y la menor brecha train-validación.

**Interpretación:** una frontera más local no produjo mejor generalización en esta grilla.

---

## 10. Árbol — profundidad

Con min_samples_leaf=50:

- profundidad 1 → validación 0,85106
- 2 → 0,88604
- 3 → 0,92116
- 4 → 0,93296
- 6 → 0,94072
- 8 → 0,94446
- 12 → 0,94442
- 20 → 0,94366
- sin límite → 0,94368

La brecha train-validación pasa de 0,0058 en profundidad 8 a ≈0,019 en profundidad 20/sin límite.

**Interpretación:**

- profundidades 1–4: underfitting;
- 6–8: mejora clara y zona útil;
- >8: train continúa mejorando mientras validación deja de hacerlo.

Este es el patrón más claro de overfitting del bloque.

---

## 11. Árbol — tamaño mínimo de hoja

Con max_depth=8:

- leaf=1 → validación 0,93476; gap 0,01720
- 5 → 0,93971
- 10 → 0,94254
- 25 → 0,94379
- 50 → 0,94446
- 100 → 0,94530
- 250 → 0,94467

Aumentar el tamaño mínimo de hoja regulariza el árbol: disminuye la brecha entre train y validación y mejora inicialmente la generalización.

---

## 12. ¿Por qué no cambiamos ahora min_samples_leaf=50 por 100?

La curva muestra un máximo de validación en 100, mientras que el GridSearch original sólo había probado:

    [1, 10, 50]

Por lo tanto, 100 no podía haber sido seleccionado en el bloque 03.

La diferencia entre 50 y 100 es:

    0,94530 - 0,94446 ≈ 0,00084 de ROC-AUC

**Decisión:** no reoptimizar retrospectivamente el modelo.

Razones:

1. el test del bloque 03 ya fue observado;
2. ampliar ahora la grilla estaría motivado por resultados posteriores;
3. la diferencia es muy pequeña;
4. sin considerar la variabilidad entre folds no corresponde presentarla como una mejora real;
5. el objetivo del bloque 04 es interpretar complejidad, no reabrir la selección.

Esto se registra como una **limitación de resolución de la grilla original**, no como un error que obligue a cambiar el modelo final.

---

## 13. Qué concluye este bloque

- Logística es muy estable en el rango de C estudiado.
- SVM generaliza mejor en la configuración regularizada seleccionada.
- El Árbol muestra con claridad underfitting a baja profundidad y overfitting cuando crece demasiado.
- Regularizar el Árbol mediante hojas más grandes reduce la brecha train-validación.
- La configuración seleccionada en el bloque 03 se encuentra en una región de buena generalización.

El bloque queda cerrado sin modificar el test ni reoptimizar modelos.
