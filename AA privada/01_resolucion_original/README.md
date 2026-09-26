# Primera forma de resolver el TP

[Comparar sus resultados con la alternativa por etapas](../COMPARACION_RESULTADOS.md).

El [informe de la primera resolución](../../Resoluci%C3%B3n%20TP%20AA/tp%20completo/TP1_completo.md)
se conserva una sola vez en `Resolución TP AA/tp completo/`. Incluye ahora
las comprobaciones que sugirió la cátedra, con resultados y gráficos dentro
del texto. Los clasificadores originales y sus resultados siguen siendo la
primera resolución; las pruebas adicionales se identifican por separado.

## Flujo de la primera forma

Partición 80/20 → EDA de entrenamiento → selección por validación cruzada de Logística,
SVM y Árbol en binario → curvas de complejidad → Voting → KNN, Árbol,
Random Forest y AdaBoost en multiclase → importancia por permutación → informe.

Los gráficos del informe están en `Resolución TP AA/tp completo/graficos/`.
El código ejecutable canónico está en `Resolución TP AA/01_entender_problema_y_datos/`
hasta `Resolución TP AA/10_resultados_discusion_informe/`; el origen de cada cifra
figura en `Resolución TP AA/tp completo/trazabilidad.md`. Los CSV generados de
`Resolución TP AA/resultados/` no están versionados; sus cifras se presentan en
los informes de cada etapa.

**Lectura principal:** el Random Forest completo depende especialmente de
aprobadas de segundo semestre, pero el experimento original no midió por separado
cuánto puede predecirse *antes* de cada semestre. La segunda forma estudia esa
pregunta con nuevas evaluaciones de la misma base, sin redefinir los resultados
del TP anterior.
