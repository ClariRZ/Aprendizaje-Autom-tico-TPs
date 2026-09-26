# 04 — ¿Qué pasa al aumentar la complejidad?

## Pregunta

¿En qué punto subir la flexibilidad mejora sólo el ajuste en entrenamiento y deja de mejorar los casos reservados dentro de train?

## Experimento

`04_curvas_complejidad.py` vuelve a crear el split **binario** del paso 03, aparta test y usa tres folds del 80 % de train. Para cada valor mide ROC-AUC en la porción usada para ajustar y en la porción retenida. Sólo mueve un hiperparámetro por curva, fijando el otro al valor del modelo ya elegido:

| Modelo | Control que se mueve | Control fijo |
|---|---|---|
| Logística | C: 0,01; 0,1; 1; 10 | Resto del pipeline del paso 03. |
| SVM | C: 0,1; 1; 10 | gamma=0,01. |
| SVM | gamma: `scale`; 0,01; 0,1 | C=0,1. |
| Árbol | profundidad: 1–20 y sin límite (valores discretos) | hoja mínima=50. |
| Árbol | hoja mínima: 1–250 (valores discretos) | profundidad=8. |

Una brecha creciente entre train y validación advierte que se ajustan peculiaridades del entrenamiento; dos puntajes bajos indican un modelo demasiado limitado. La curva **no** cambia la selección tras mirar test. Por ejemplo, profundidad 8 con hoja 50 es el Árbol binario seleccionado; si alguna hoja vecina luce apenas mejor en esta exploración, no se sustituye a posteriori.

## Script, archivos y lugar en el informe

```bash
python "Resolución TP AA/04_curvas_complejidad_y_generalizacion/04_curvas_complejidad.py"
```

Escribe `Resolución TP AA/resultados/04_complejidad/04_curvas_complejidad.csv` y cinco PNG en `graficos/`. La figura 3 y la sección 3.3 del [informe final](../../entrega_final/TP1.md) resumen estas curvas. La figura 4 añade curvas **multiclase** calculadas por los experimentos complementarios de `AA privada/03_verificaciones_catedra/`; este script, por sí solo, reproduce la parte binaria.

