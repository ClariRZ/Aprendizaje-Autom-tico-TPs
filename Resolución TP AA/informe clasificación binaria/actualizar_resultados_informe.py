from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parents[1]
ORIGEN = BASE / "resultados"
DESTINO = Path(__file__).resolve().parent / "resultados"

ARCHIVOS = [
    "03_binario.csv",
    "03_cv_busqueda.csv",
]

DESTINO.mkdir(parents=True, exist_ok=True)

for nombre in ARCHIVOS:
    src = ORIGEN / nombre
    dst = DESTINO / nombre
    if not src.exists():
        print(f"FALTA -> {src}")
        continue
    shutil.copy2(src, dst)
    print(f"OK -> {dst}")
