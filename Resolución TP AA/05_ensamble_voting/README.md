# 05 — ¿Ayuda combinar los tres modelos binarios?

## Pregunta

¿Promediar las probabilidades de Logística, SVM y Árbol mejora a cada uno por separado en la **misma tarea binaria**?

## Decisiones y explicación

`05_voting.py` usa la misma división binaria 80/20 del paso 03 y conserva sus parámetros elegidos: Logística C=0,1; SVM C=0,1 y gamma=0,01; Árbol profundidad 8 y hoja 50. Soft Voting promedia probabilidades con pesos iguales. El SVM individual no entrega probabilidades; `CalibratedClassifierCV` envuelve el pipeline completo de preprocesamiento y SVM y aprende una sigmoide con tres folds internos de entrenamiento (`ensemble=False`). Así también imputadores, codificación y escalado se reajustan dentro de cada fold de calibración. El promedio asigna la clase de mayor probabilidad. Ni los pesos ni la calibración se ajustaron mirando test; el test había sido observado previamente y sus resultados se consideran exploratorios.

Probamos una mejora posible de ROC-AUC, no asumimos que deba mejorar también el F1: ordenar mejor todos los casos puede coexistir con una sensibilidad menor a un umbral fijo. Los tres modelos observan las mismas variables, por lo que algunos errores serán compartidos y el promedio no podrá borrarlos.

## Script y salida

```bash
python "Resolución TP AA/05_ensamble_voting/05_voting.py"
```

Escribe `Resolución TP AA/resultados/05_voting.csv` con métricas y conteos de confusión, más `05_comparacion_voting.csv`. Este segundo archivo incorpora los resultados del paso 03 **si ya existe** `03_binario.csv`; conviene ejecutar el paso 03 antes. En la tabla 2 del [informe final](../../entrega_final/TP1.md), Voting alcanza AUC 0,9514 frente a 0,9506 de la Logística, pero F1 0,8472 frente a 0,8510; la sección 3.2 explica por qué una sola cifra no agota la comparación.

