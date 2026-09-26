# Informe — Comparación final de modelos

## 1. Clasificación binaria

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Voting soft | 0,9053 | 0,9080 | 0,7940 | 0,8472 | **0,95135** |
| Logística | **0,9066** | 0,9004 | **0,8067** | **0,8510** | 0,95059 |
| SVM | 0,8979 | **0,9150** | 0,7619 | 0,8315 | 0,94598 |
| Árbol | 0,9057 | 0,9064 | 0,7970 | 0,8482 | 0,94401 |

Voting obtiene el mayor ROC-AUC, pero la mejora frente a Logística es mínima. La Regresión Logística mantiene el mejor perfil en accuracy, recall y F1. SVM prioriza precision a costa de recall.

Los gaps CV-test de Logística, SVM y Árbol fueron 0,000104, 0,000185 y 0,000446, respectivamente, lo que muestra una gran estabilidad.

## 2. Clasificación multiclase

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | ROC-AUC macro OvR |
|---|---:|---:|---:|---:|---:|
| Random Forest | **0,8196** | **0,7896** | **0,7750** | **0,7808** | **0,9254** |
| AdaBoost | 0,8182 | 0,7874 | 0,7707 | 0,7776 | 0,9219 |
| Árbol | 0,8106 | 0,7774 | 0,7647 | 0,7699 | 0,9170 |
| KNN | 0,7943 | 0,7664 | 0,7388 | 0,7483 | 0,9034 |

Random Forest presenta el mejor balance global. Los dos ensambles mejoran al Árbol individual. En Curso continúa siendo la clase más difícil.

## 3. Lectura integrada

La tarea binaria alcanza una capacidad discriminativa muy alta con modelos clásicos; el Voting sólo añade una mejora marginal de ROC-AUC.

La tarea multiclase es más exigente porque debe separar Graduado, En Curso y Desertor. En ese contexto, los ensambles aportan una mejora moderada y Random Forest ofrece el mejor equilibrio global.

Las curvas de complejidad muestran que más complejidad no implica mejor generalización, y el análisis de importancia indica que la señal predictiva se concentra principalmente en variables académicas y, en segundo lugar, administrativas/económicas.

## 4. Hipótesis

- H1 — Actividad/rendimiento académico: fuertemente respaldada.
- H2 — Inactividad/ceros: compatible, pero no aislada de H1.
- H3 — Administrativo/económico: respaldada de forma heterogénea.
- H4 — Señales de ingreso: débilmente respaldada.

## 5. Cierre

El bloque 09 queda cerrado. El paso siguiente es redactar el informe final integrando métodos, resultados, discusión, limitaciones y conclusión.
