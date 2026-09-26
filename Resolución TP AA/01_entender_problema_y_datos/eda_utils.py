import numpy as np
import pandas as pd

CLASES = ["Graduado", "En Curso", "Desertor"]

ACADEMICAS = [
    "Unidades_curriculares_1er_semestre_(inscrito)",
    "Cantidad_de_evaluaciones_en_el_1er_semestre",
    "Unidades_curriculares_1er_semestre_(aprobadas)",
    "Nota_promedio_en_el_1er_semestre",
    "Unidades_curriculares_2do_semestre_(inscrito)",
    "Cantidad_de_evaluaciones_en_el_2do_semestre_",
    "Unidades_curriculares_2do_semestre_(aprobadas)",
    "Nota_promedio_en_el_2do_semestre",
]

ACADEMICAS_CLAVE = [
    "Unidades_curriculares_1er_semestre_(aprobadas)",
    "Nota_promedio_en_el_1er_semestre",
    "Unidades_curriculares_2do_semestre_(aprobadas)",
    "Nota_promedio_en_el_2do_semestre",
]

ADMIN_CLAVE = [
    "Pago_de_matrícula_al_día",
    "Tiene_deuda",
    "Poseedor_de_Beca",
    "sexo",
]

LABELS = {
    "Unidades_curriculares_1er_semestre_(inscrito)": "Inscriptas S1",
    "Cantidad_de_evaluaciones_en_el_1er_semestre": "Evaluaciones S1",
    "Unidades_curriculares_1er_semestre_(aprobadas)": "Aprobadas S1",
    "Nota_promedio_en_el_1er_semestre": "Nota S1",
    "Unidades_curriculares_2do_semestre_(inscrito)": "Inscriptas S2",
    "Cantidad_de_evaluaciones_en_el_2do_semestre_": "Evaluaciones S2",
    "Unidades_curriculares_2do_semestre_(aprobadas)": "Aprobadas S2",
    "Nota_promedio_en_el_2do_semestre": "Nota S2",
    "Edad_al_momento_de_la_inscripción": "Edad",
    "Puntaje_en_examen_de_ingreso": "Puntaje ingreso",
    "Cualificación_promedio_de_estudios_previos": "Nota previa",
    "Pago_de_matrícula_al_día": "Matrícula al día",
    "Tiene_deuda": "Deuda",
    "Poseedor_de_Beca": "Beca",
    "modo_aplicacion": "Modo de aplicación",
    "Carrera": "Carrera",
    "Estudios_máximos_antes_de_la_inscripción": "Estudios previos",
    "estado_civil": "Estado civil",
    "Asistencia_diurna/nocturna": "Turno",
    "Necesidades_educativas_especiales": "Necesidades educativas",
    "Estudiante_Internacional": "Internacional",
    "desplazado": "Desplazado",
    "Demanda_Cog_Padre": "Demanda cognitiva padre",
    "Demanda_Cog_Madre": "Demanda cognitiva madre",
    "Ingresos_Padre": "Ingresos padre",
    "Ingresos_Madre": "Ingresos madre",
}

def etiqueta(c):
    return LABELS.get(c, c)

SIN_FECHA = {
    "Tiene_deuda", "Pago_de_matrícula_al_día", "Poseedor_de_Beca",
    "Demanda_Cog_Padre", "Demanda_Cog_Madre", "Ingresos_Padre", "Ingresos_Madre",
}

def momento(c):
    if "1er_semestre" in c:
        return "1er semestre"
    if "2do_semestre" in c:
        return "2do semestre"
    if c in SIN_FECHA:
        return "fecha sin acreditar"
    return "ingreso candidato"

def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    obs = tab.to_numpy(dtype=float)
    n = obs.sum()
    if n == 0:
        return np.nan
    expected = obs.sum(axis=1, keepdims=True) @ obs.sum(axis=0, keepdims=True) / n
    mask = expected > 0
    chi2 = (((obs - expected) ** 2) / np.where(mask, expected, 1))[mask].sum()
    denom = min(obs.shape[0] - 1, obs.shape[1] - 1)
    return np.sqrt((chi2 / n) / denom) if denom > 0 else np.nan

def eta_squared(x, grupos):
    tmp = pd.DataFrame({"x": x, "g": grupos}).dropna()
    if tmp.empty:
        return np.nan
    media = tmp["x"].mean()
    ss_total = ((tmp["x"] - media) ** 2).sum()
    if ss_total == 0:
        return np.nan
    ss_between = 0.0
    for _, s in tmp.groupby("g")["x"]:
        ss_between += len(s) * (s.mean() - media) ** 2
    return ss_between / ss_total

def smd_binario(x, y):
    x0 = x[y == 0].dropna()
    x1 = x[y == 1].dropna()
    pooled = np.sqrt((x0.var(ddof=1) + x1.var(ddof=1)) / 2)
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return (x1.mean() - x0.mean()) / pooled
