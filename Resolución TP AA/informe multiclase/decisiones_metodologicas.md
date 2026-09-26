# Decisiones metodológicas — Clasificación multiclase

## 1. Target original

**Decisión:** volver a utilizar las tres clases originales:

- Graduado;
- En Curso;
- Desertor.

**Por qué:** la segunda etapa de la consigna solicita clasificación multiclase y el EDA mostró que En Curso no debe asumirse idéntico a Graduado.

---

## 2. Nueva optimización

**Decisión:** no reutilizar automáticamente los hiperparámetros del Árbol binario.

**Por qué:** cambió la tarea, cambió el target y cambió la métrica de selección.

La optimización multiclase se realiza de nuevo usando únicamente train.

---

## 3. Split

**Decisión:** 80% train / 20% test, estratificado y con semilla 42.

Distribución observada:

### Train
- Desertor: 0,3306
- En Curso: 0,1953
- Graduado: 0,4742

### Test
- Desertor: 0,3306
- En Curso: 0,1953
- Graduado: 0,4741

**Por qué:** preserva prácticamente de manera exacta la composición de las tres clases.

---

## 4. Modelos

Se comparan:

- K-NN;
- Árbol de Decisión.

**Por qué:** son los algoritmos indicados por la consigna para la etapa multiclase.

Ambos admiten multiclase de manera nativa, por lo que no es necesario forzar One-vs-One u One-vs-Rest.

---

## 5. Preprocesamiento de KNN

**Decisión:** estandarizar variables numéricas y aplicar One-Hot a categóricas.

**Por qué:** KNN depende de distancias. Sin escalado, variables con rangos numéricos grandes podrían dominar artificialmente la distancia.

---

## 6. Preprocesamiento del Árbol

**Decisión:** no estandarizar variables numéricas.

**Por qué:** los árboles utilizan puntos de corte y no dependen de distancias o magnitudes comparables entre variables.

---

## 7. Métrica de selección: F1 macro

**Decisión:** GridSearchCV optimiza F1 macro.

**Por qué:** las clases tienen tamaños diferentes. F1 macro calcula F1 por clase y luego promedia otorgando el mismo peso a Graduado, En Curso y Desertor.

Esto evita que Graduado, la clase más grande, domine el criterio de selección.

---

## 8. Grilla de KNN

Se probaron:

    n_neighbors = [3, 5, 9, 15, 25]
    weights = ["uniform", "distance"]

Resultado seleccionado:

    n_neighbors = 25
    weights = "uniform"

La mejor configuración está en el extremo superior de la grilla para `n_neighbors`.

**Decisión:** no ampliar ahora la grilla después de observar test.

**Interpretación:** la búsqueda sugiere que un vecindario relativamente amplio generaliza mejor dentro de los valores probados, pero no demuestra que 25 sea el óptimo global.

---

## 9. Grilla del Árbol

Se probaron:

    max_depth = [2, 4, 6, 8, 12, None]
    min_samples_leaf = [1, 10, 50]

Resultado seleccionado:

    max_depth = 12
    min_samples_leaf = 50

`min_samples_leaf=50` está en el extremo superior de la grilla.

**Decisión:** no ampliar retrospectivamente la búsqueda después de observar test.

Esto se registra como una limitación de resolución de la grilla.

---

## 10. Métricas finales

Se reportan:

- accuracy;
- precision macro;
- recall macro;
- F1 macro;
- ROC-AUC macro One-vs-Rest;
- precision, recall y F1 por clase.

**Por qué:** una única métrica global puede ocultar que un modelo funciona muy bien para Graduado pero mal para En Curso.

---

## 11. Estabilidad CV vs test

### KNN
- F1 macro CV: 0,748694
- F1 macro test: 0,748318
- diferencia ≈ 0,00038

### Árbol
- F1 macro CV: 0,769756
- F1 macro test: 0,769926
- diferencia ≈ -0,00017

**Interpretación:** ambos modelos muestran una estabilidad muy alta entre validación cruzada y test para esta partición.

---

## 12. No comparar directamente binario y multiclase como si fueran la misma tarea

El bloque binario tuvo ROC-AUC alrededor de 0,95.

El multiclase alcanza:

- KNN: ROC-AUC macro OvR 0,9034
- Árbol: 0,9170

**Decisión:** no interpretar esta diferencia como una caída directa de “calidad” porque cambió la tarea, el target y la definición de las métricas.

La clasificación en tres clases es un problema más exigente y contiene una clase En Curso que antes estaba fusionada con Graduado.

---

## 13. Test no se reutiliza para ajustar

Después de observar estos resultados:

- no se amplía la grilla;
- no se cambian hiperparámetros;
- no se modifica F1 macro como scoring.

El test sigue siendo una evaluación final de esta búsqueda.

---

## 14. Cierre

El objetivo de este bloque es comparar KNN y Árbol en la tarea multiclase y estudiar cómo se distribuyen los errores entre las tres clases.

Los resultados son suficientes para cerrar el bloque sin perseguir mejoras marginales post-hoc.
