# Decisiones metodológicas — Bagging y Boosting

## 1. Objetivo

Comparar dos estrategias de ensamble basadas en árboles:

- **Random Forest** como método de Bagging;
- **AdaBoost** como método de Boosting.

La pregunta es:

> ¿los ensambles mejoran la capacidad de generalización respecto de un Árbol individual en la tarea multiclase?

---

## 2. Misma tarea y mismo test que en el bloque 06

Se mantiene:

- target = Graduado / En Curso / Desertor;
- split 80/20;
- estratificación;
- semilla 42.

**Por qué:** la comparación con el Árbol debe hacerse sobre los mismos casos.

---

## 3. Nueva búsqueda de hiperparámetros

**Decisión:** cada ensamble se ajusta mediante GridSearchCV usando únicamente train.

**Por qué:** Random Forest y AdaBoost son modelos nuevos; no sería metodológicamente correcto fijar una única configuración arbitraria.

---

## 4. Métrica de selección

Se mantiene **F1 macro**.

**Por qué:** da el mismo peso a las tres clases y evita que Graduado, la clase más numerosa, domine la optimización.

---

## 5. Random Forest

Grilla:

    n_estimators = [200, 500]
    max_depth = [12, None]
    min_samples_leaf = [1, 50]
    max_features = ["sqrt"]

Mejor configuración:

    n_estimators = 500
    max_depth = None
    min_samples_leaf = 1
    max_features = "sqrt"

### Interpretación

Random Forest permite árboles individuales profundos, pero reduce varianza al combinar muchos árboles entrenados con bootstrap y selección aleatoria de variables.

Que la mejor configuración use `max_depth=None` y `min_samples_leaf=1` no implica automáticamente overfitting del ensamble completo: el mecanismo de bagging modifica la relación sesgo-varianza respecto de un Árbol aislado.

---

## 6. AdaBoost

Grilla:

    estimator__max_depth = [1, 2]
    n_estimators = [100, 200, 400]
    learning_rate = [0.05, 0.1]

Mejor configuración:

    estimator__max_depth = 2
    n_estimators = 400
    learning_rate = 0.1

AdaBoost construye clasificadores secuencialmente, concentrándose cada vez más en observaciones mal clasificadas.

---

## 7. Hiperparámetros en el borde de la grilla

En Random Forest, `n_estimators=500` es el máximo probado.

En AdaBoost, los tres valores seleccionados se encuentran en el extremo superior de sus respectivos rangos:

- depth = 2;
- n_estimators = 400;
- learning_rate = 0.1.

**Decisión:** no ampliar las grillas después de haber observado test.

**Por qué:**

1. el test ya fue consultado;
2. ampliar ahora la búsqueda introduciría optimización post-hoc;
3. el objetivo del TP es comparar modelos y comprender sus propiedades, no maximizar indefinidamente una métrica;
4. los resultados ya son estables entre CV y test.

Se registra como una limitación de resolución de las grillas.

---

## 8. Estabilidad CV vs test

### Random Forest

- F1 macro CV = 0,781374
- F1 macro test = 0,780761
- diferencia ≈ 0,00061

### AdaBoost

- F1 macro CV = 0,771512
- F1 macro test = 0,777557
- diferencia ≈ -0,00605

Ambos muestran buena estabilidad; Random Forest es especialmente consistente entre CV y test.

---

## 9. Comparación con Árbol individual

### Árbol
- F1 macro = 0,769926
- ROC-AUC macro OvR = 0,916961

### Random Forest
- F1 macro = 0,780761
- ROC-AUC macro OvR = 0,925385

### AdaBoost
- F1 macro = 0,777557
- ROC-AUC macro OvR = 0,921888

Los dos ensambles mejoran al Árbol individual en las métricas globales.

---

## 10. Diferencias por clase

### Desertor
- Árbol F1 = 0,8472
- Random Forest F1 = 0,8529
- AdaBoost F1 = 0,8562

AdaBoost obtiene el mejor F1 para Desertor.

### En Curso
- Árbol F1 = 0,5899
- Random Forest F1 = 0,6096
- AdaBoost F1 = 0,6006

Random Forest mejora más la clase más difícil.

### Graduado
- Árbol F1 = 0,8726
- Random Forest F1 = 0,8798
- AdaBoost F1 = 0,8758

Random Forest también obtiene el mejor F1 para Graduado.

---

## 11. Costo computacional

- Random Forest: ≈ 0,74 min
- AdaBoost: ≈ 2,38 min

AdaBoost fue aproximadamente tres veces más lento en esta búsqueda.

El tiempo se interpreta como característica práctica de esta implementación y hardware, no como métrica de calidad predictiva.

---

## 12. Cierre

Los ensambles aportan una mejora real pero moderada respecto del Árbol individual.

Random Forest presenta el mejor equilibrio global.

AdaBoost destaca levemente en F1 de Desertor.

No se reabren las grillas después de observar test.

Bloque 07 cerrado.
