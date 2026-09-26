# 01 — Entender la base y formular las preguntas

## Pregunta

¿Qué representan las 27 variables, cómo se distribuyen y qué diferencias observables aparecen entre Desertor, En Curso y Graduado antes de entrenar modelos?

## Datos y secuencia

Primero reconstruí `00_base_de_datos/dataset_TP1_2026.csv` con `python 00_base_de_datos/reconstruir_dataset.py`. Desde la raíz del repositorio ejecutá:

```bash
python "Resolución TP AA/01_entender_problema_y_datos/00_correr_eda.py"
```

`_comun.cargar_eda_train()` reserva 10.000 de las 50.000 filas como test, con semilla 42 y estratificación por **tres** clases. Todos los cálculos exploratorios usan las 40.000 restantes. El ejecutor borra antes `Resolución TP AA/resultados/eda_exploratorio/` para evitar archivos obsoletos.

| Script | Pregunta concreta y salida |
|---|---|
| `01_panorama_y_calidad.py` | Tamaños, clases, diccionario observado, faltantes, repetidos e inconsistencias. Tablas 01–04. |
| `02_univariado.py` | ¿Cómo es **cada** variable sin mirar aún el target? Rangos, medianas, ceros, categorías y grupos raros. Tablas 05–07. |
| `03_clases_y_target.py` | ¿Cómo cambia cada variable entre las tres clases? Compara numéricas, categorías y tasas; separa ceros académicos de rendimiento entre quienes tienen valor positivo. Incluye puente binario descriptivo. Tablas 08–14. |
| `04_relaciones_entre_predictores.py` | ¿Qué predictores repiten información? Correlación de Spearman y V de Cramér entre pares. Tablas 15–18. |
| `05_graficos_finales.py` | Dibuja distribución de clases, variables, ceros, notas, administración y redundancia. Diez PNG en `graficos/`. |
| `06_informe_final_eda.py` | Une tablas y figuras en `EDA_FINAL_PARA_EVALUAR.md`, con preguntas abiertas para la interpretación. |

`eda_utils.py` guarda nombres de variables y funciones de asociación usadas por estos scripts. Los gráficos son una ayuda para responder preguntas, no una votación automática de variables “importantes”.

## Decisiones y lectura

- Se conservaron filas idénticas y casos con más aprobadas que inscriptas: no hay identificador de estudiante ni definiciones institucionales para corregirlos sin inventar datos. La sensibilidad a perfiles repetidos aparece en el informe final, sección 3.7.
- Un cero en aprobadas o nota puede significar varias cosas; el código mide **por separado** cuánto hay de ceros y cómo se distribuyen los valores positivos. No convierte ceros en faltantes.
- Tasas por categoría deben leerse junto con su `n`: un grupo pequeño puede dar una tasa llamativa poco estable. Spearman, V de Cramér y otras medidas describen asociación, no causalidad ni utilidad segura dentro de un modelo.
- El EDA llevó a **H1** (actividad y rendimiento académico), **H2** (información administrativa y de ingreso) y **H3** (dificultad de reconocer En Curso). La formulación y los datos centrales están en [el informe editable](../../entrega_final/TP1.md), secciones 1 y 3.1.

El EDA **no eliminó variables**: los modelos completos usan los 27 predictores. Una exploración anterior a esta separación había observado el test; el informe presenta las métricas de test como exploratorias, no como confirmación independiente.

