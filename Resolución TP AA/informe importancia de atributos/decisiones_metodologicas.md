# Decisiones metodológicas — Importancia de atributos

## 1. Modelo interpretado

**Decisión:** interpretar exactamente el Random Forest seleccionado en el bloque 07:

    n_estimators = 500
    max_depth = None
    min_samples_leaf = 1
    max_features = "sqrt"

**Por qué:** la importancia debe corresponder al modelo que realmente evaluamos como mejor ensamble multiclase.

---

## 2. Método: Permutation Importance

**Decisión:** usar Permutation Importance sobre las 27 variables originales.

**Por qué:** mide cuánto cae F1 macro cuando se rompe una variable, lo que produce una interpretación directamente vinculada a performance predictiva.

Además, al operar sobre el Pipeline completo, una variable categórica se permuta como una unidad y no como múltiples dummies del One-Hot Encoding.

---

## 3. Test reservado

**Decisión:** no usar el test para calcular importancia.

La importancia se calculó dentro del 80% de train mediante 3 folds estratificados.

En cada fold:

1. se ajustó Random Forest en 2/3 de train;
2. se evaluó F1 macro en el tercio retenido;
3. se permutó cada variable cinco veces.

**Por qué:** evita reutilizar test para interpretar variables después de haberlo usado ya para evaluar generalización.

---

## 4. Repeticiones

Cada variable tiene:

    3 folds × 5 repeticiones = 15 estimaciones

Se reportan media y desvío estándar.

**Por qué:** una única permutación puede depender del azar. Las repeticiones permiten cuantificar estabilidad.

---

## 5. Métrica de importancia

**Decisión:** medir caída de **F1 macro**.

**Por qué:** es la misma métrica usada para seleccionar los modelos multiclase y da el mismo peso a las tres clases.

---

## 6. Importancia individual vs bloques

Las variables se agruparon conceptualmente en:

- Académico;
- Administrativo/económico;
- Ingreso/contexto.

**Cautela:** la suma de importancias positivas por bloque no debe interpretarse como una descomposición exacta de la performance.

Los bloques tienen distinto número de variables y existen correlaciones fuertes entre predictores.

Por eso se usan como orientación descriptiva, no como una partición matemática del F1.

---

## 7. Correlación entre variables

El EDA mostró alta correlación entre:

- aprobadas S1 y S2;
- notas S1 y S2;
- inscriptas S1 y S2.

Permutation Importance de una variable puede ser menor cuando otra variable correlacionada conserva información similar.

Por eso:

> importancia individual baja no implica ausencia de importancia del fenómeno subyacente.

---

## 8. Contraste de hipótesis

### H1 — Actividad y rendimiento académico

**Resultado:** fuertemente respaldada.

El bloque académico concentra las mayores caídas de F1 macro.

### H2 — Inactividad académica

**Resultado:** compatible, pero no separada directamente.

Las variables con alta prevalencia de ceros son importantes, pero el método permuta la variable completa. No distingue cuánto de esa importancia proviene de los ceros y cuánto del nivel entre valores positivos.

No se afirma que H2 quede demostrada de forma independiente.

### H3 — Administrativo/económico

**Resultado:** respaldada de forma heterogénea.

- Matrícula al día: importancia alta.
- Beca: importancia moderada.
- Deuda: importancia baja pero positiva.

### H4 — Señales de ingreso

**Resultado:** respaldo débil.

Edad tiene una pequeña importancia positiva. Puntaje de ingreso es muy pequeño y su desvío es mayor que la media.

No corresponde afirmar una capacidad predictiva fuerte de este bloque.

---

## 9. Variables con importancia cercana a cero o negativa

Permutation Importance puede producir valores cercanos a cero o ligeramente negativos.

Eso significa que, dentro de la variabilidad de las repeticiones, romper esa variable no deterioró de forma consistente F1 macro.

No significa que la variable sea “protectora” ni que tenga efecto causal negativo.

---

## 10. No usar el ranking para reentrenar

**Decisión:** no eliminar variables y volver a entrenar modelos a partir de este ranking.

**Por qué:** el test ya fue observado en los bloques previos y eso abriría una nueva ronda de ajuste post-hoc.

El ranking se usa para interpretación y discusión, no para una nueva selección de features.

---

## 11. Cierre

Este bloque responde qué variables aportan información predictiva al Random Forest seleccionado y cómo esas señales se relacionan con las hipótesis del EDA.

No responde causalidad ni efecto de intervención.

Bloque 08 cerrado.
