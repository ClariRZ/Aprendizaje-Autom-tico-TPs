# TP1 · Aprendizaje Automático

## Empezar por la versión final

La [resolución final editable](entrega_final/TP1.md) es la versión para
entregar. Incluye una precisión sobre el efecto pequeño de excluir perfiles
repetidos. Las figuras están junto al Markdown para poder corregir texto,
tablas y referencias directamente en Git. La lista de integrantes figura
como `[completar]` y debe completarse antes de entregar.

## Material de trabajo

- [Primera resolución y código del TP](Resoluci%C3%B3n%20TP%20AA/README.md): copia de la
  resolución compartida, con EDA, preprocesamiento, modelos y comprobaciones.
- [EDA explicado y comparación por momentos](AA%20privada/README.md): desarrollo
  adicional, hipótesis sobre las 27 variables, alternativa temporal y pruebas
  sugeridas por la cátedra. Sus resultados se presentan como experimentos
  complementarios; el informe de `entrega_final/` es la versión para entregar.
- [`00_base_de_datos/`](00_base_de_datos/): diccionario, partes de la base y
  programa de reconstrucción necesarios para ejecutar los scripts.

## Resolución paso a paso

Cada carpeta contiene el código de ese paso y un `README.md` que explica qué
pregunta se responde, por qué se tomó cada decisión y cómo se refleja en la
entrega final. Seguí este orden:

| Paso | Pregunta que responde |
|---|---|
| [01 · Datos y EDA](Resoluci%C3%B3n%20TP%20AA/01_entender_problema_y_datos/README.md) | ¿Qué significan las variables y qué señales aparecen en train? |
| [02 · Preprocesamiento](Resoluci%C3%B3n%20TP%20AA/02_preprocesamiento_sin_fugas/README.md) | ¿Cómo preparar cada fold sin filtrar información? |
| [03 · Modelos binarios](Resoluci%C3%B3n%20TP%20AA/03_clasificacion_binaria/README.md) | ¿Cómo distinguir Desertor de No desertor? |
| [04 · Complejidad](Resoluci%C3%B3n%20TP%20AA/04_curvas_complejidad_y_generalizacion/README.md) | ¿Qué muestran train y validación al cambiar la complejidad? |
| [05 · Voting](Resoluci%C3%B3n%20TP%20AA/05_ensamble_voting/README.md) | ¿Mejora la combinación de modelos binarios? |
| [06 · Tres clases](Resoluci%C3%B3n%20TP%20AA/06_clasificacion_multiclase/README.md) | ¿Qué ocurre al distinguir Desertor, En Curso y Graduado? |
| [07 · Ensambles](Resoluci%C3%B3n%20TP%20AA/07_bagging_y_boosting/README.md) | ¿Cómo se comparan Random Forest y AdaBoost? |
| [08 · Importancia](Resoluci%C3%B3n%20TP%20AA/08_importancia_de_variables/README.md) | ¿De qué variables depende el bosque? |
| [09 · Comparación](Resoluci%C3%B3n%20TP%20AA/09_comparacion_final/README.md) | ¿Qué resultados sostienen las hipótesis? |
| [10 · Informe](Resoluci%C3%B3n%20TP%20AA/10_resultados_discusion_informe/README.md) | ¿Cómo se convierten resultados y límites en la entrega? |

Los ensayos adicionales pedidos por la cátedra tienen su código y resultados
en [`AA privada/03_verificaciones_catedra/`](AA%20privada/03_verificaciones_catedra/README.md).
La comparación por momentos de información está en
[`AA privada/02_alternativa_temporal/`](AA%20privada/02_alternativa_temporal/README.md).

## Reproducir el trabajo

Desde la raíz, instalá las dependencias con
`python -m pip install -r requirements.txt` y reconstruí el CSV siguiendo
[`00_base_de_datos/README.md`](00_base_de_datos/README.md). El EDA y el ajuste
de modelos se hicieron sobre train; test se reservó para evaluar. El archivo
final distingue los resultados principales de las verificaciones adicionales.

La estructura `Resolución TP AA/` y `AA privada/` conserva las rutas que usan
los scripts y los enlaces de los informes. La copia repetida del informe y sus
gráficos dentro de `AA privada/01_resolucion_original/` se sustituyó por una
guía que apunta a la fuente única en `Resolución TP AA/tp completo/`.
