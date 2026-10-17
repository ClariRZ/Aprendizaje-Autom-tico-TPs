# Decisiones metodológicas — Voting

## 1. Soft Voting

**Decisión:** usar Soft Voting.

**Por qué:** permite combinar la información continua de probabilidad de cada modelo, en lugar de usar sólo votos de clase.

---

## 2. Pesos iguales

**Decisión:** cada modelo recibe el mismo peso.

**Por qué:** no existe un criterio externo definido para asignar pesos y optimizarlos con test introduciría ajuste al conjunto de prueba.

---

## 3. Modelos base congelados

**Decisión:** reutilizar exactamente los hiperparámetros seleccionados en el bloque 03.

No se reabre GridSearch ni se cambia el Árbol a partir de las curvas del bloque 04.

---

## 4. Calibración del SVM

Soft Voting requiere `predict_proba`.

Se usa `CalibratedClassifierCV` con calibración sigmoide y tres folds internos (`ensemble=False`). Su estimador es el pipeline completo de preprocesamiento y SVM, de modo que los imputadores, el codificador y el escalador se ajustan sólo con la parte de entrenamiento de cada fold interno.

**Por qué:** convierte el score del SVM en probabilidades sin cambiar C ni gamma.

**Importante:** se corrigió una versión previa que ajustaba el preprocesamiento antes de las divisiones internas de calibración. Se repitió el Voting conservando los mismos hiperparámetros, pesos y split, sin usar el test para elegirlos. La interpretación y las métricas deben basarse en esa nueva corrida. El test sigue siendo exploratorio porque ya había sido observado.

---

## 5. Mismo test

Se mantiene exactamente el split del bloque 03.

**Por qué:** una comparación justa exige evaluar todos los modelos sobre los mismos casos.

---

## 6. No elegir por una sola métrica

Voting obtuvo una leve mejora en ROC-AUC pero no en F1/recall.

**Decisión:** no llamarlo superior de manera global.

Cada métrica responde a una pregunta distinta y todavía no existe un costo explícito de falsos positivos/falsos negativos que defina un único criterio operativo.

---

## 7. No optimizar pesos después de mirar test

Podríamos intentar dar más peso a Logística, pero hacerlo ahora usando estos resultados sería ajuste retrospectivo.

Por eso no se hace.

---

## 8. Cierre

El objetivo del bloque es evaluar la utilidad de combinar modelos, no maximizar indefinidamente una métrica.

Una mejora marginal de AUC junto con resultados similares o peores en otras métricas es un resultado válido y suficiente para cerrar Voting.
