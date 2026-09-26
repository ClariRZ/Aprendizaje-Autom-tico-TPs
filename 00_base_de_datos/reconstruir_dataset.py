from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
PARTES = sorted((BASE / "dataset_partes").glob("dataset_parte_*.csv"))

N_PARTES = 10
FILAS_POR_PARTE = 5000
N_COLUMNAS = 28
FILAS_TOTALES = 50000

if len(PARTES) != N_PARTES:
    raise RuntimeError(f"Se esperaban {N_PARTES} partes y hay {len(PARTES)}")

dfs = []
columnas = None

for parte in PARTES:
    df_parte = pd.read_csv(parte)

    if df_parte.shape != (FILAS_POR_PARTE, N_COLUMNAS):
        raise RuntimeError(
            f"{parte.name}: se esperaba {(FILAS_POR_PARTE, N_COLUMNAS)} "
            f"y se obtuvo {df_parte.shape}"
        )

    if columnas is None:
        columnas = list(df_parte.columns)
    elif list(df_parte.columns) != columnas:
        raise RuntimeError(f"{parte.name}: el encabezado no coincide con las otras partes")

    dfs.append(df_parte)

df = pd.concat(dfs, ignore_index=True)

if df.shape != (FILAS_TOTALES, N_COLUMNAS):
    raise RuntimeError(
        f"Dataset reconstruido: se esperaba {(FILAS_TOTALES, N_COLUMNAS)} "
        f"y se obtuvo {df.shape}"
    )

salida = BASE / "dataset_TP1_2026.csv"
df.to_csv(salida, index=False, encoding="utf-8-sig")
print(f"OK: {df.shape} -> {salida}")
