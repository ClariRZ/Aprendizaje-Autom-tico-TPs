-- Adapta TP1.md al formato de artículo (tipo MDPI) al convertir a LaTeX.
-- * Imagen + párrafo "**Figura N.** ..." -> figura con epígrafe debajo (versión PDF).
-- * Párrafo "**Tabla N.** ..." + tabla -> tabla con epígrafe arriba.
-- * Párrafo inmediatamente posterior a una tabla -> nota en letra pequeña.
-- Los rótulos se escriben en el texto, por eso los epígrafes no se numeran.

local function empieza_con(bloque, prefijo)
  if not bloque or bloque.t ~= "Para" then return false end
  local primero = bloque.content[1]
  return primero and primero.t == "Strong" and pandoc.utils.stringify(primero):match("^" .. prefijo .. " %d+%.") ~= nil
end

local function latex(inlines)
  return pandoc.write(pandoc.Pandoc({ pandoc.Plain(inlines) }), "latex"):gsub("%s+$", "")
end

local function imagen(bloque)
  local img
  bloque:walk({ Image = function(i) img = i end })
  return img
end


-- Convierte la longtable de pandoc en un tabular dentro de un flotante
-- (las tablas no se cortan entre páginas, como en el formato de artículo).
local function tabular(tbl)
  local tex = pandoc.write(pandoc.Pandoc({ tbl }), "latex")
  tex = tex:gsub("\\begin{longtable}%[%]", "\\begin{tabular*}{\\linewidth}")
  tex = tex:gsub("{@{}", "{@{\\extracolsep{\\fill}}", 1)
  tex = tex:gsub("\\end{longtable}", "\\bottomrule\n\\end{tabular*}")
  tex = tex:gsub("\\endfirsthead.-\\endhead\n", "")
  tex = tex:gsub("\\endhead\n", "")
  tex = tex:gsub("\\bottomrule\\noalign{}\n\\endlastfoot\n", "")
  tex = tex:gsub("\\bottomrule\\noalign{}\n\\endfoot\n", "")
  return tex
end

function Pandoc(doc)
  local entrada, salida = doc.blocks, {}
  local i = 1
  while i <= #entrada do
    local b, sig = entrada[i], entrada[i + 1]
    local img = (b.t == "Figure" or b.t == "Para") and imagen(b) or nil
    if img and empieza_con(sig, "Figura") then
      local src = img.src:gsub("%.svg$", ".pdf")
      table.insert(salida, pandoc.RawBlock("latex",
        "\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=\\linewidth]{" .. src .. "}\n" ..
        "\\caption*{" .. latex(sig.content) .. "}\n\\end{figure}"))
      i = i + 2
    elseif empieza_con(b, "Tabla") and sig and sig.t == "Table" then
      local tex = "\\begin{table}[!htbp]\n\\begin{tabla}\n\\epigrafetabla{" .. latex(b.content) .. "}\n" ..
        tabular(sig)
      i = i + 2
      local nota = entrada[i]
      if nota and nota.t == "Para" then
        tex = tex .. "\n\\notatabla{" .. latex(nota.content) .. "}"
        i = i + 1
      end
      table.insert(salida, pandoc.RawBlock("latex", tex .. "\n\\end{tabla}\n\\end{table}"))
    else
      table.insert(salida, b)
      i = i + 1
    end
  end
  doc.blocks = salida
  return doc
end
