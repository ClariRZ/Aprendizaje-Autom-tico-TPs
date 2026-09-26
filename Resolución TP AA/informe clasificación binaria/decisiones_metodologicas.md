# Decisiones metodológicas — Clasificación binaria

## 1. Definición de la tarea

**Decisión:** definir una tarea binaria con:

- Desertor = 1;
- Graduado + En Curso = 0.

**Por qué:** la consigna solicita comenzar con clasificación binaria. Se conserva Desertor como clase positiva porque es el evento que se busca detectar.

**Implicancia:** “No desertor” es una clase heterogénea, porque el EDA mostró diferencias entre Graduado y En Curso. Esto debe recordarse al interpretar resultados.

---

## 2. Conservación de los 27 predictores

**Decisión:** no eliminar predictores a partir del EDA.

**Por qué:** el EDA describe asociaciones; no demuestra utilidad predictiva. Seleccionar variables sólo porque aparecieron asociadas al target podría convertir una exploración descriptiva en un filtro guiado por el resultado.

**Alternativa descartada:** entrenar únicamente con las variables mejor rankeadas en el EDA.

**Qué se hará después:** la importancia predictiva se estudiará en el bloque específico de importancia de atributos.

---

## 3. Split train/test

**Decisión:** 80% train y 20% test.

- train: 40.000 casos;
- test: 10.000 casos.

**Por qué:** reserva una muestra suficientemente grande para estimar generalización sin reducir demasiado el conjunto disponible para ajuste y validación.

---

## 4. Estratificación

**Decisión:** usar `stratify=y`.

**Resultado:**

- prevalencia total de Desertor: 0,33058;
- train: 0,330575;
- test: 0,3306.

**Por qué:** evita que una diferencia aleatoria en la proporción de clases contamine la comparación entre entrenamiento y prueba.

---

## 5. Semilla

**Decisión:** usar semilla 42.

**Por qué:** hace reproducible la partición y permite comparar los modelos sobre exactamente los mismos casos.

**Importante:** la semilla no se eligió porque produzca una mejor performance.

---

## 6. Preprocesamiento dentro del Pipeline

**Decisión:** mantener las transformaciones dentro de un `Pipeline`.

### Numéricas
- imputación por mediana;
- estandarización para Logística y SVM.

### Categóricas
- imputación por moda;
- One-Hot Encoding.

### Árbol
- no se escala la parte numérica.

**Por qué:** Logística y SVM son sensibles a la escala; los árboles trabajan mediante puntos de corte y no requieren estandarización.

**Control de leakage:** durante la validación cruzada, cada fold aprende sus transformaciones únicamente desde el subconjunto de entrenamiento de ese fold.

---

## 7. No aplicar remuestreo de clases inicialmente

**Decisión:** no usar oversampling, undersampling ni SMOTE.

**Por qué:** Desertor representa aproximadamente 33% de la muestra, por lo que el desbalance no es extremo. Se optó primero por conservar la distribución real y evaluar con múltiples métricas.

**Alternativa:** reconsiderar remuestreo sólo si aparecieran problemas claros de sensibilidad hacia la clase positiva.

---

## 8. Modelos elegidos

**Decisión:** comparar:

1. Regresión Logística;
2. SVM con kernel RBF;
3. Árbol de Decisión.

**Por qué:** son los modelos solicitados en la consigna para la primera etapa binaria y representan familias con comportamientos distintos:

- Logística: modelo lineal probabilístico;
- SVM RBF: frontera no lineal;
- Árbol: reglas jerárquicas y no lineales.

---

## 9. Criterio de selección de hiperparámetros

**Decisión:** usar ROC-AUC como `scoring` de GridSearchCV.

**Por qué:** ROC-AUC evalúa la capacidad de ordenar casos positivos por encima de negativos a través de todos los umbrales, sin fijar todavía un punto de corte operativo específico.

**Importante:** no elegimos hiperparámetros mirando accuracy, F1 o test después de conocer los resultados.

---

## 10. Validación cruzada

**Decisión:** StratifiedKFold de 3 folds, con shuffle y semilla 42.

**Por qué:**

- conserva la proporción de clases en cada fold;
- permite comparar hiperparámetros sobre particiones repetidas de train;
- mantiene un costo computacional razonable para SVM RBF en 40.000 casos.

**Alternativa descartada por costo:** aumentar mucho el número de folds en esta etapa, especialmente porque cada combinación de SVM es costosa.

---

## 11. Grilla de Regresión Logística

Se probaron:

    C = [0.01, 0.1, 1, 10]

**Qué controla C:** la fuerza de regularización; valores pequeños implican mayor regularización.

**Seleccionado:** C = 0,1.

**Lectura:** la mejor validación apareció con una solución regularizada, no con el valor de C más alto.

---

## 12. Grilla de SVM RBF

Se probaron:

    C = [0.1, 1, 10]
    gamma = ["scale", 0.01, 0.1]

**Qué controla C:** penalización de errores / regularización.

**Qué controla gamma:** alcance de influencia de cada observación en el kernel RBF; valores grandes producen fronteras más locales y potencialmente más complejas.

**Seleccionado:**

    C = 0,1
    gamma = 0,01

**Lectura:** también se seleccionó una configuración relativamente regularizada.

---

## 13. Grilla del Árbol

Se probaron:

    max_depth = [2, 4, 6, 8, 12, None]
    min_samples_leaf = [1, 10, 50]

**Qué controla max_depth:** cantidad máxima de niveles del árbol.

**Qué controla min_samples_leaf:** cantidad mínima de observaciones en una hoja.

**Seleccionado:**

    max_depth = 8
    min_samples_leaf = 50

**Lectura:** el mejor resultado de CV no corresponde al árbol sin límite ni a hojas mínimas de 1 caso, sino a una solución regularizada.

---

## 14. Uso del conjunto de test

**Decisión:** test se utiliza una sola vez, después de seleccionar los hiperparámetros usando únicamente train + CV.

**Por qué:** si se modificaran modelos después de observar test, test dejaría de funcionar como estimación independiente de generalización.

**Regla desde ahora:** no cambiar hiperparámetros del bloque 03 porque alguna métrica de test resulte más atractiva.

---

## 15. Métricas reportadas

Se reportan:

- accuracy;
- precision;
- recall;
- F1;
- ROC-AUC.

**Por qué usar varias:** cada métrica describe un aspecto distinto del error.

- **Accuracy:** proporción total de clasificaciones correctas.
- **Precision:** entre quienes el modelo predice como Desertor, cuántos realmente lo son.
- **Recall:** entre los verdaderos Desertores, cuántos son detectados.
- **F1:** equilibrio entre precision y recall.
- **ROC-AUC:** capacidad global de discriminación a través de diferentes umbrales.

No se define todavía un costo diferencial de falsos positivos vs falsos negativos, por lo que no se selecciona un modelo mediante una regla económica o clínica.

---

## 16. Decisiones computacionales que NO cambian el experimento

Para acelerar la ejecución:

- se limitó GridSearch a un máximo de 4 procesos;
- se cacheó temporalmente el preprocesamiento;
- SVC usa cache de kernel.

**Por qué:** el SVM RBF era el cuello de botella.

**Importante:** estas modificaciones no cambian datos, grillas, folds, scoring ni criterio de selección. Son optimizaciones de ejecución, no decisiones estadísticas.

---

## 17. Qué NO podemos concluir todavía

Este bloque permite comparar capacidad predictiva global de tres modelos.

Todavía no permite afirmar:

- qué variable “causa” deserción;
- qué hipótesis del EDA está confirmada individualmente;
- qué variable es la más importante;
- qué modelo se comportará mejor en otra institución;
- que una diferencia pequeña de AUC sea estadísticamente significativa.

Esas preguntas requieren los bloques posteriores de importancia, complejidad, discusión y limitaciones.
