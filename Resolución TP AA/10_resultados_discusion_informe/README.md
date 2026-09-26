# 10 — Pasar de tablas a un informe defendible

## Pregunta

¿Qué observamos, cómo respaldan o debilitan los datos H1–H3 y hasta dónde se pueden extender las conclusiones?

## Documento final

La resolución adoptada para leer, corregir y editar está en [entrega_final/TP1.md](../../entrega_final/TP1.md), el informe final editable. Sus secciones siguen el orden: introducción e hipótesis; datos, tareas y método; EDA; resultados binarios; complejidad y controles; multiclase; disponibilidad temporal; sensibilidad a repetidos; importancia; discusión; limitaciones y conclusión. Mantiene tablas Markdown y figuras SVG editables en `entrega_final/figuras/`.

La discusión debe distinguir tres tipos de evidencia:

- **EDA sobre train:** muestra asociaciones y ayuda a formular H1 (académico), H2 (administrativo e ingreso) y H3 (En Curso).
- **Validación sobre train:** el contraste de bloques temporales, los folds, Dummy, curvas e importancia prueban cuánto cambia la predicción al cambiar información o complejidad.
- **Test ya observado históricamente:** describe los resultados finales y la matriz de confusión, pero no constituye una validación independiente nueva de hipótesis desarrolladas después de verlo.

La conclusión del informe habla del **primer año completo** cuando usa las 27 variables. El aumento de desempeño al añadir S1 motiva estudiar una alerta temprana futura, pero las siete variables sin fecha no se asignan a una etapa temprana. En Curso es un estado sin desenlace registrado y no se interpreta como futura graduación o deserción.

## Script auxiliar existente

```bash
python "Resolución TP AA/10_resultados_discusion_informe/10_borrador.py"
```

Produce `Resolución TP AA/resultados/borrador_resultados.md` a partir de tablas previas **si existen**. Es un borrador automático de presentación, no genera ni sustituye el informe elegido: cuenta la distribución de clases sobre la base completa de 50.000, mientras el EDA final y sus porcentajes del informe se basan en 40.000 de train. Para cifras de las hipótesis y texto final, consultá `entrega_final/TP1.md` y los resultados originales de cada paso. La figura 4 (curvas multiclase), figura 5 (Dummy y folds) y tabla 3 (grillas ampliadas) usan `AA privada/03_verificaciones_catedra/`; la figura 7 (bloques por momento de información) usa `AA privada/02_alternativa_temporal/modelos/comparar_etapas.py`. Ningún script de esta carpeta reproduce por sí solo todas las figuras del informe final.
