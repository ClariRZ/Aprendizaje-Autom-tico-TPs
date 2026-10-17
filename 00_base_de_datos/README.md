# 00 — Base de datos

La base utilizada en el TP tiene **50.000 registros y 28 columnas: 27 predictores y `target`**. Los nombres y categorías están en castellano. Se guarda en 10 fragmentos de 5.000 filas, todos con el mismo encabezado.

Ejecutar:

```bash
python reconstruir_dataset.py
```

Esto genera `dataset_TP1_2026.csv` con exactamente 50.000 filas de datos.

También se incluye un diccionario resumido. La base utilizada no se modifica durante el análisis.

## Generación y reconstrucción son procedimientos distintos

[`generar_dataset_tp1.py`](generar_dataset_tp1.py) es la copia sin cambios
del generador adjuntado para verificar la preparación. Requiere un
`train.csv` de Kaggle como entrada; programa traducciones, agrupamientos,
reescalados, la exclusión de los códigos de carrera 39 y 979 y variables
parentales derivadas de ocupación. Estas últimas son recodificaciones
definidas por el script, no mediciones directas de ingresos o cognición.
El objetivo predeterminado es 50.000 filas y la semilla predeterminada es
42. Muestrea sin reemplazo por respuesta y carrera y ajusta el total al
final por el redondeo de los estratos. Esos valores pueden cambiarse con
`--filas` y `--semilla`.

`reconstruir_dataset.py` sólo concatena las diez partes ya preparadas; no
traduce, no reescala ni genera observaciones sintéticas.

La revisión confirmó que el CSV adjuntado tiene exactamente los mismos
datos que el reconstruido y el mismo orden de columnas que la salida del
generador. Su SHA-256 es
`a0a75e40e32ae336c24fad49fb271da45429593e1fccc16f08c397a1f3bbe09d`.
El generador tiene SHA-256
`3662e401085ba0c172a9f9edbe918d3835bdc7345f97ae4bfacdc80f76198193`.

No se contó en esta revisión con el `train.csv` de entrada ni con un
registro de la ejecución que produjo el archivo final. Por eso se
documentan las operaciones programadas y la base observada, sin afirmar
que se haya reproducido de extremo a extremo su generación, ni atribuir
su autoría o ejecución a la cátedra o a integrantes concretos del grupo.
