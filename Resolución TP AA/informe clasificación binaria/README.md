# Informe — Clasificación binaria

Esta carpeta documenta el bloque 03 del TP.

Se separan deliberadamente tres cosas:

- **informe_clasificacion_binaria.md**: resultados, comparación e interpretación;
- **decisiones_metodologicas.md**: qué decisiones tomamos, por qué y qué alternativas descartamos;
- **resultados/**: snapshot versionado de los CSV que respaldan el informe.

El código reproducible está en:

    ../03_clasificacion_binaria/

Los resultados técnicos se generan primero en:

    ../resultados/03_binario.csv
    ../resultados/03_cv_busqueda.csv

Como `../resultados/` está ignorado por Git, los resultados que forman parte del informe se copian luego a:

    resultados/03_binario.csv
    resultados/03_cv_busqueda.csv

con:

    python "Resolución TP AA/informe clasificación binaria/actualizar_resultados_informe.py"

## Estado

El bloque 03 queda cerrado una vez revisados:
- selección de hiperparámetros;
- consistencia entre CV y test;
- métricas finales;
- decisiones metodológicas;
- límites de interpretación.

El paso siguiente es estudiar curvas de complejidad y generalización.
