"""Compila TP1.md en un PDF con formato de artículo (tipo MDPI).

Desde la raíz del repositorio:
    python entrega_final/scripts/generar_figuras.py   # genera también las figuras en PDF
    python entrega_final/pdf/compilar_pdf.py

Requiere pandoc (>= 3) y xelatex. Escribe el proyecto LaTeX editable en
entrega_final/latex/ (TP1.tex, figuras/, latexmkrc) y lo compila; con
--zip genera además TP1_latex.zip para subir a Overleaf. El texto, las tablas y los epígrafes se
toman de TP1.md sin modificarlo: título, integrantes, resumen y palabras
clave pasan a la portada; el resto conserva el orden del Markdown.
"""
from pathlib import Path
import re
import subprocess
import tempfile

import shutil
import sys

import yaml

BASE = Path(__file__).resolve().parents[1]
PDF = Path(__file__).resolve().parent
md = (BASE / "TP1.md").read_text(encoding="utf-8")

titulo = re.search(r"^# (.+)$", md, re.M).group(1).strip()
curso = re.search(r"^\*\*(Trabajo práctico.+?)\*\*", md, re.M).group(1).strip()
autores = re.search(r"^\*\*Integrantes:\*\*\s*(.+)$", md, re.M).group(1).strip()
resumen = re.search(r"^## Resumen\s*\n(.+?)\n(?=## )", md, re.M | re.S).group(1)
clave = re.search(r"^\*\*Palabras clave:\*\*\s*(.+)$", resumen, re.M)
palabras = clave.group(1).strip() if clave else ""
resumen = resumen[:clave.start()].strip() if clave else resumen.strip()
cuerpo = md[md.index("## 1."):]

meta = {"title": titulo, "curso": curso, "curso-corto": "Aprendizaje Automático 2026 · TP1",
        "autores": autores, "abstract": resumen, "palabras-clave": palabras}
fuente = "---\n" + yaml.safe_dump(meta, allow_unicode=True, width=10_000) + "---\n\n" + cuerpo

TEX = BASE / "latex"  # proyecto editable (p. ej., para Overleaf)
TEX.mkdir(exist_ok=True)
(TEX / "figuras").mkdir(exist_ok=True)
for fig in (BASE / "figuras").glob("figura_*.pdf"):
    shutil.copy2(fig, TEX / "figuras" / fig.name)
(TEX / "latexmkrc").write_text("$pdf_mode = 5;  # compilar con XeLaTeX\n", encoding="utf-8")

with tempfile.TemporaryDirectory() as tmp:
    entrada = Path(tmp) / "TP1.md"
    entrada.write_text(fuente, encoding="utf-8")
    subprocess.run([
        "pandoc", str(entrada), "-o", str(TEX / "TP1.tex"), "--standalone",
        "--from", "markdown+pipe_tables-implicit_figures",
        "--template", str(PDF / "plantilla.tex"),
        "--lua-filter", str(PDF / "filtro.lua"),
        "--columns", "150", "--shift-heading-level-by=-1",
    ], check=True)

for _ in range(2):  # dos pasadas para "página N de M"
    subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "TP1.tex"],
                   cwd=TEX, check=True, stdout=subprocess.DEVNULL)
shutil.copy2(TEX / "TP1.pdf", PDF / "TP1.pdf")
for aux in TEX.glob("TP1.*"):
    if aux.suffix not in {".tex"}:
        aux.unlink()
if "--zip" in sys.argv:
    shutil.make_archive(str(BASE / "TP1_latex"), "zip", TEX)
    print(BASE / "TP1_latex.zip")
print(TEX / "TP1.tex", PDF / "TP1.pdf", sep="\n")
