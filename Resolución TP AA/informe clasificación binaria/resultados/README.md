# Resultados versionados del bloque 03

Esta carpeta guarda **copias estables de los resultados que se citan en el informe**.

Los archivos técnicos originales se generan en:

```text
Resolución TP AA/resultados/
```

Esa carpeta está ignorada por Git porque contiene salidas regenerables.

## Qué se versiona acá

- `03_binario.csv`: métricas finales de los modelos elegidos.
- `03_cv_busqueda.csv`: candidatos de GridSearchCV, media/desvío de ROC-AUC y ranking.

La idea es conservar en Git exactamente la evidencia usada para redactar el informe, sin versionar todas las salidas temporales del pipeline.

## Actualización

Después de ejecutar el bloque 03:

```bash
python "Resolución TP AA/informe clasificación binaria/actualizar_resultados_informe.py"
```

El script copia los dos CSV desde `../resultados/` a esta carpeta.
