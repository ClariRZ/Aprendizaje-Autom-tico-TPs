# Decisiones metodológicas — Comparación final

## 1. Mantener separadas las tareas

La clasificación binaria y la multiclase no se comparan como si fueran el mismo problema. Cambian el target, la dificultad y las métricas.

## 2. Mismo test dentro de cada tarea

Los modelos de una misma tarea se comparan sobre exactamente los mismos casos de test.

## 3. Métricas

En binario se reportan accuracy, precision, recall, F1 y ROC-AUC.

En multiclase, F1 macro es la métrica central porque da igual peso a Graduado, En Curso y Desertor; también se reporta ROC-AUC macro OvR.

## 4. No declarar un ganador universal

Voting logra el mayor ROC-AUC binario, pero Logística presenta mayor recall y F1.

Random Forest logra el mejor equilibrio global multiclase, mientras AdaBoost presenta el mejor F1 específico para Desertor.

## 5. Generalización

Los gaps CV-test son pequeños en los modelos con validación comparable, lo que respalda estabilidad dentro de esta partición.

## 6. No reabrir hiperparámetros

Este bloque integra resultados ya cerrados. No se reentrena, no se amplían grillas y no se reutiliza test para nuevas decisiones.
