"""Compila TP1.md en un PDF con formato de artículo (tipo MDPI).

Desde la raíz del repositorio:
    python entrega_final/scripts/generar_figuras.py   # genera también las figuras en PDF
    python entrega_final/pdf/compilar_pdf.py

Requiere pandoc (>= 3) y xelatex. El texto, las tablas y los epígrafes se
toman de TP1.md sin modificarlo: título, integrantes, resumen y palabras
clave pasan a la portada; el resto conserva el orden del Markdown.
"""
from pathlib import Path
import re
import subprocess
import tempfile

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

with tempfile.TemporaryDirectory() as tmp:
    entrada = Path(tmp) / "TP1.md"
    entrada.write_text(fuente, encoding="utf-8")
    subprocess.run([
        "pandoc", str(entrada), "-o", str(PDF / "TP1.pdf"),
        "--from", "markdown+pipe_tables-implicit_figures",
        "--template", str(PDF / "plantilla.tex"),
        "--lua-filter", str(PDF / "filtro.lua"),
        "--pdf-engine", "xelatex", "--columns", "150", "--shift-heading-level-by=-1",
        "--resource-path", str(BASE),
    ], check=True, cwd=BASE)
print(PDF / "TP1.pdf")
