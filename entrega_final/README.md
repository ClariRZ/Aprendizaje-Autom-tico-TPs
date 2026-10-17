# Entrega final

[`TP1.md`](TP1.md) es el informe final editable en Git y la única fuente del
texto. [`pdf/TP1.pdf`](pdf/TP1.pdf) es la versión maquetada con formato de
artículo (inspirado en *Applied Sciences*, MDPI): siete tablas de tres
líneas con el mejor valor en negrita y siete figuras con paneles (a), (b)…
descritos en el epígrafe.

Antes de entregar, completá `Integrantes: [completar]` con los nombres del
grupo, regenerá el PDF y revisá que tablas, cifras y figuras coincidan con
los resultados.

## Regenerar figuras y PDF

Desde la raíz del repositorio:

```bash
python 00_base_de_datos/reconstruir_dataset.py
python entrega_final/scripts/predicciones_test.py          # reajusta los 8 modelos finales
python entrega_final/scripts/cv_configuraciones_finales.py # media ± DE de validación (Tablas 3 y 4)
python entrega_final/scripts/generar_figuras.py            # figuras en SVG (Markdown) y PDF (LaTeX)
python entrega_final/pdf/compilar_pdf.py                   # requiere pandoc >= 3 y xelatex
```

`predicciones_test.py` comprueba que las métricas reproduzcan
`verificaciones/metricas_reproducidas.csv` al redondear a cuatro decimales
y guarda las salidas de prueba en `datos_figuras/`. Ningún script elige
configuraciones: solo describen los modelos ya seleccionados.

- `scripts/`: cálculo de salidas de prueba, validación cruzada y figuras.
- `datos_figuras/`: datos de las figuras y de las columnas de validación.
- `pdf/`: plantilla LaTeX, filtro de pandoc y PDF compilado.
- `verificaciones/`: evidencias de reproducción y de la corrección del Voting.

Las [evidencias de verificación](verificaciones/README.md) registran la
reproducción de las métricas y la nueva corrida del Voting con el
preprocesamiento dentro de sus folds internos de calibración.

El [recorrido en diez pasos](../README.md#resolución-paso-a-paso) reúne los
scripts y las decisiones que fundamentan el informe. Los experimentos
adicionales y la comparación temporal están identificados en `AA privada/`.
