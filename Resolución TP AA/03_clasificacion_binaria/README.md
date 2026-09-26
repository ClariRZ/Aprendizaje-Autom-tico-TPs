# 03 — Primera tarea: Desertor frente a No desertor

## Pregunta

Si juntamos En Curso y Graduado en **No desertor**, ¿qué tan bien detectan Desertor tres formas distintas de clasificar?

## Decisiones

El positivo es Desertor (`1`). Con `train_test_split(test_size=0.20, stratify=y, random_state=42)` se reservan 10.000 filas y se entrenan modelos en 40.000. Aquí `stratify` usa la **etiqueta binaria**; por eso este test difiere del test multiclase y del EDA.

Probamos Regresión Logística, SVM RBF y Árbol porque capturan relaciones lineales, fronteras no lineales y reglas de división, respectivamente. Cada uno incluye su preprocesamiento en el mismo `Pipeline`. `GridSearchCV` prueba una grilla pequeña con tres folds estratificados **sólo en train** y elige el mayor ROC-AUC: esta métrica ordena los casos según su probabilidad o puntaje sin fijar un umbral. Se probaron C de la Logística (0,01; 0,1; 1; 10); C y gamma del SVM (3×3 combinaciones), y profundidad y tamaño mínimo de hoja del Árbol (6×3).

La grilla seleccionó C=0,1 para Logística; C=0,1 y gamma=0,01 para SVM; profundidad 8 y hoja mínima 50 para Árbol. El mejor se reajusta con todo train y **luego** se evalúa en test. Precisión, sensibilidad y F1 usan las etiquetas predichas; Logística y Árbol usan umbral 0,5 para la clase positiva y SVM individual, el signo de su función de decisión. La ROC-AUC usa puntajes continuos.

## Ejecutar y comprobar

```bash
python "Resolución TP AA/03_clasificacion_binaria/03_modelos_binarios.py"
```

Lee `00_base_de_datos/dataset_TP1_2026.csv` y genera en `Resolución TP AA/resultados/`:

- `03_binario.csv`: métricas de test, mejor CV, tiempo y parámetros elegidos.
- `03_cv_busqueda.csv`: puntajes medios, dispersión y orden de **todos** los candidatos ensayados.

En el [informe final](../../entrega_final/TP1.md), tabla 2 y sección 3.2, Logística obtiene ROC-AUC 0,9506 y F1 0,8510. Es una evaluación exploratoria porque una fase temprana del trabajo ya había consultado el test. El test no se usa para volver a elegir la grilla.

