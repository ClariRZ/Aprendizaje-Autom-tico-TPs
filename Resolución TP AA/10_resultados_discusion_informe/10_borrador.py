from pathlib import Path
import sys,pandas as pd
R=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(R))
from _comun import cargar,OUT
df=cargar()
parts=[f"# Borrador de resultados\n\nMuestra: {len(df):,} estudiantes; {df.shape[1]-1} predictores.\n",
"## Distribución del target\n"+df.target.value_counts().to_markdown()]
for f,t in [("09_comparacion_binaria.csv","## Clasificación binaria"),("09_comparacion_multiclase.csv","## Multiclase"),("08_importancia.csv","## Importancia predictiva")]:
 p=OUT/f
 if p.exists(): parts.append(t+"\n"+pd.read_csv(p).head(15).to_markdown(index=False))
(OUT/"borrador_resultados.md").write_text("\n\n".join(parts),encoding="utf-8")
print("OK",OUT/"borrador_resultados.md")
