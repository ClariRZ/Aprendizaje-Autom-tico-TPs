# 00 — Base de datos

La base original tiene 50.000 filas. Para poder versionarla de manera estable mediante el conector se guarda en 10 fragmentos de 5.000 filas, todos con el mismo encabezado.

Ejecutar:

```bash
python reconstruir_dataset.py
```

Esto genera `dataset_TP1_2026.csv` con exactamente 50.000 filas de datos.

También se incluye un diccionario resumido. La base original no se modifica durante el análisis.
