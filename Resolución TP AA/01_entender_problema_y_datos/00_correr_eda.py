from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parent
RES = BASE.parent / "resultados" / "eda_exploratorio"

if RES.exists():
    shutil.rmtree(RES)

scripts = [
    BASE / "01_panorama_y_calidad.py",
    BASE / "02_univariado.py",
    BASE / "03_clases_y_target.py",
    BASE / "04_relaciones_entre_predictores.py",
    BASE / "05_graficos_finales.py",
    BASE / "06_informe_final_eda.py",
]

for script in scripts:
    print("\n" + "=" * 80)
    print(script.name)
    print("=" * 80)
    subprocess.run([sys.executable, str(script)], check=True)

print("\nEDA FINAL SOLO SOBRE TRAIN")
print("Revisá:")
print(RES / "EDA_FINAL_PARA_EVALUAR.md")
print(RES / "graficos")
