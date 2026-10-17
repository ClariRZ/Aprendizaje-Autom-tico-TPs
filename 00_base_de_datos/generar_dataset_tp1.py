#!/usr/bin/env python3
"""Genera el dataset final del TP1 a partir del dataset aumentado de Kaggle.

Este script toma el archivo ``train.csv`` de la competencia de Kaggle
"Playground Series - S4E6" (una versión aumentada sintéticamente del
dataset original de UCI "Predict students' dropout and academic success"),
y lo transforma en el dataset que usa la cátedra:

    1. Traduce y renombra las variables al español, con el mismo estilo
       de nombres de columna que ya usa la cátedra.
    2. Traduce/agrupa las variables categóricas crudas (que en Kaggle son
       códigos numéricos) a las categorías en texto que usamos nosotros,
       usando siempre un criterio documentado (traducción directa cuando
       existe una correspondencia 1 a 1, o un estándar internacional
       -ISCED para nivel educativo, ISCO-08 para ocupación- cuando hace
       falta agrupar códigos).
    3. Reescala las notas a la convención numérica de la cátedra.
    4. Sub-muestrea el dataset a una cantidad fija de filas, de forma
       estratificada (por carrera y por resultado académico) para
       mantener las proporciones del dataset aumentado original.

El proceso es reproducible: fijando la semilla aleatoria (parámetro
--semilla) se obtiene siempre exactamente el mismo resultado.

Uso:
    python generar_dataset_tp1.py --entrada train.csv \
        --salida dataset_TP1_2026.csv

Requisitos: pandas, numpy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Parámetros generales
# --------------------------------------------------------------------------- #

SEMILLA_POR_DEFECTO: int = 42
FILAS_OBJETIVO_POR_DEFECTO: int = 50_000

# Códigos de "Course" que aparecen con una sola fila en el dataset de Kaggle.
# Son ruido de la generación sintética, no carreras reales, así que se
# descartan antes de cualquier otra transformación.
CODIGOS_CARRERA_RUIDO: List[int] = [39, 979]

# --------------------------------------------------------------------------- #
# Diccionarios de mapeo (código crudo de UCI/Kaggle -> valor de la cátedra)
# --------------------------------------------------------------------------- #

# Estado civil: traducción directa 1 a 1 de los códigos oficiales de UCI.
MAPA_ESTADO_CIVIL: Dict[int, str] = {
    1: "Soltero",
    2: "Casado",
    3: "Viudo",
    4: "Divorciado",
    5: "Unión de hecho",
    6: "Separado legalmente",
}

# Carrera ("Course"): traducción directa 1 a 1 de las 17 carreras reales de
# UCI/Kaggle. Los únicos dos códigos que se fusionan son los que representan
# el mismo programa dictado en turno diurno y nocturno (Gestión y Servicio
# Social); esa distinción ya se captura aparte en "Asistencia_diurna/nocturna",
# así que no hace falta duplicar la carrera para eso.
MAPA_CARRERA: Dict[int, str] = {
    33: "Tecnología en Producción de Biocombustibles",
    171: "Animación y Diseño Multimedia",
    8014: "Servicio Social",
    9238: "Servicio Social",
    9003: "Agronomía",
    9070: "Diseño de Comunicación",
    9085: "Enfermería Veterinaria",
    9119: "Ingeniería Informática",
    9130: "Equinocultura",
    9147: "Gestión",
    9991: "Gestión",
    9254: "Turismo",
    9500: "Enfermería",
    9556: "Higiene Bucodental",
    9670: "Publicidad y Marketing",
    9773: "Periodismo y Comunicación",
    9853: "Educación Básica",
}

# Nivel educativo máximo antes de la inscripción ("Previous qualification"):
# se agrupan los 21 códigos originales según el nivel ISCED (International
# Standard Classification of Education, UNESCO) al que corresponden.
MAPA_ESTUDIOS_MAXIMOS: Dict[int, str] = {
    # ISCED 1-2: ciclos básicos / secundario no completado.
    9: "Educación Básica",
    10: "Educación Básica",
    12: "Educación Básica",
    14: "Educación Básica",
    15: "Educación Básica",
    19: "Educación Básica",
    38: "Educación Básica",
    # ISCED 3: secundario completo.
    1: "Educación Secundaria",
    # ISCED 5-6: técnico, licenciatura, grado.
    2: "Superior - Pregrado",
    3: "Superior - Pregrado",
    6: "Superior - Pregrado",
    39: "Superior - Pregrado",
    40: "Superior - Pregrado",
    42: "Superior - Pregrado",
    # ISCED 7-8: maestría, doctorado.
    4: "Superior - Posgrado",
    5: "Superior - Posgrado",
    43: "Superior - Posgrado",
    # Códigos rarísimos (menos de 10 filas en total) sin nivel claro.
    11: "Otros / No completado",
    17: "Otros / No completado",
    36: "Otros / No completado",
    37: "Otros / No completado",
}

# Modo de aplicación/admisión ("Application mode"): no existe un estándar
# internacional para esto (es específico del sistema portugués), así que se
# agrupa por el criterio que describe cada código oficial de UCI.
MAPA_MODO_APLICACION: Dict[int, str] = {
    # Contingente general, fases 1ª/2ª/3ª.
    1: "Acceso General",
    17: "Acceso General",
    18: "Acceso General",
    # Contingente especial regional (islas Azores y Madeira).
    5: "Contingente Especial (Regional)",
    16: "Contingente Especial (Regional)",
    # Poseedores de otro curso superior (ya tienen un título).
    7: "Acceso para Titulados",
    # Todo lo demás: ordenanzas específicas, mayores de 23 años, cambio de
    # carrera/institución, estudiante internacional, diplomas técnicos, etc.
    2: "Acceso por Normativa Específica",
    3: "Acceso por Normativa Específica",
    4: "Acceso por Normativa Específica",
    9: "Acceso por Normativa Específica",
    10: "Acceso por Normativa Específica",
    12: "Acceso por Normativa Específica",
    15: "Acceso por Normativa Específica",
    26: "Acceso por Normativa Específica",
    27: "Acceso por Normativa Específica",
    35: "Acceso por Normativa Específica",
    39: "Acceso por Normativa Específica",
    42: "Acceso por Normativa Específica",
    43: "Acceso por Normativa Específica",
    44: "Acceso por Normativa Específica",
    51: "Acceso por Normativa Específica",
    53: "Acceso por Normativa Específica",
}

# Ocupación de padres -> grupo ocupacional ISCO-08 (primer dígito del código).
# Los códigos 90 y 99 ("otra situación" / no especificado) se tratan como
# el grupo de menor calificación, por ser la opción más conservadora.
_CODIGOS_ISCO_SIN_GRUPO_CLARO = (90, 99)

# A partir del grupo ISCO-08 se derivan dos escalas ordinales de 5 niveles:
# demanda cognitiva del puesto e ingresos estimados asociados.
MAPA_DEMANDA_COGNITIVA_POR_GRUPO_ISCO: Dict[int, str] = {
    0: "Muy baja demanda cognitiva",  # Fuerzas Armadas / sin grupo claro
    1: "Muy alta demanda cognitiva",  # Directores y gerentes
    2: "Muy alta demanda cognitiva",  # Profesionales científicos
    3: "Alta demanda cognitiva",  # Técnicos y profesionales de nivel medio
    4: "Demanda cognitiva media",  # Personal de apoyo administrativo
    5: "Demanda cognitiva media",  # Trabajadores de servicios y ventas
    6: "Demanda cognitiva baja",  # Agropecuarios calificados
    7: "Demanda cognitiva baja",  # Oficiales, operarios y artesanos
    8: "Muy baja demanda cognitiva",  # Operadores de instalaciones y máquinas
    9: "Muy baja demanda cognitiva",  # Ocupaciones elementales
}

MAPA_INGRESOS_POR_GRUPO_ISCO: Dict[int, str] = {
    0: "Bajos ingresos",
    1: "Altos ingresos",
    2: "Altos ingresos",
    3: "Ingresos medios-altos",
    4: "Ingresos medios",
    5: "Ingresos medios",
    6: "Ingresos medios-bajos",
    7: "Ingresos medios-bajos",
    8: "Bajos ingresos",
    9: "Bajos ingresos",
}

# Situación académica final ("Target"): traducción directa.
MAPA_TARGET: Dict[str, str] = {
    "Dropout": "Desertor",
    "Enrolled": "En Curso",
    "Graduate": "Graduado",
}

# Variables binarias que solo necesitan traducir Sí/No o el par de categorías.
MAPA_SEXO: Dict[int, str] = {0: "Femenino", 1: "Masculino"}
MAPA_SI_NO: Dict[int, str] = {0: "No", 1: "Sí"}
MAPA_TURNO: Dict[int, str] = {0: "Nocturna", 1: "Diurna"}

# Orden final de columnas, con el mismo estilo de nombres que usa la cátedra
# (sin el sufijo "_label": el nombre de la variable ya deja en claro que es
# una categoría).
ORDEN_COLUMNAS_FINAL: List[str] = [
    "Estudios_máximos_antes_de_la_inscripción",
    "estado_civil",
    "modo_aplicacion",
    "sexo",
    "desplazado",
    "target",
    "Carrera",
    "Asistencia_diurna/nocturna",
    "Cualificación_promedio_de_estudios_previos",
    "Puntaje_en_examen_de_ingreso",
    "Necesidades_educativas_especiales",
    "Tiene_deuda",
    "Pago_de_matrícula_al_día",
    "Poseedor_de_Beca",
    "Edad_al_momento_de_la_inscripción",
    "Estudiante_Internacional",
    "Unidades_curriculares_1er_semestre_(inscrito)",
    "Cantidad_de_evaluaciones_en_el_1er_semestre",
    "Unidades_curriculares_1er_semestre_(aprobadas)",
    "Nota_promedio_en_el_1er_semestre",
    "Unidades_curriculares_2do_semestre_(inscrito)",
    "Cantidad_de_evaluaciones_en_el_2do_semestre_",
    "Unidades_curriculares_2do_semestre_(aprobadas)",
    "Nota_promedio_en_el_2do_semestre",
    "Demanda_Cog_Padre",
    "Demanda_Cog_Madre",
    "Ingresos_Padre",
    "Ingresos_Madre",
]


# --------------------------------------------------------------------------- #
# Funciones auxiliares
# --------------------------------------------------------------------------- #

def obtener_grupo_isco(codigo_ocupacion: int) -> int:
    """Devuelve el grupo ocupacional ISCO-08 (0-9) de un código de ocupación.

    En la codificación de UCI, los códigos de un solo dígito (0-9) son
    directamente el grupo ISCO-08 mayor. Los códigos de más dígitos son
    subgrupos ISCO-08 más finos, cuyo primer dígito indica el grupo mayor
    (por ejemplo, 122 pertenece al grupo 1). Los códigos 90 y 99
    ("otra situación" / no especificado) no tienen grupo ISCO real, así que
    se tratan como el grupo de menor calificación (0).

    Parameters
    ----------
    codigo_ocupacion:
        Código crudo de "Mother's occupation" / "Father's occupation".

    Returns
    -------
    int
        Grupo ocupacional ISCO-08 mayor, entre 0 y 9.
    """
    codigo_ocupacion = int(codigo_ocupacion)
    if codigo_ocupacion in _CODIGOS_ISCO_SIN_GRUPO_CLARO:
        return 0
    return int(str(codigo_ocupacion)[0])


def cargar_datos_crudos(ruta_train_csv: Path) -> pd.DataFrame:
    """Carga el train.csv de Kaggle y descarta las filas de ruido.

    Las filas de ruido son las que tienen un código de "Course" que aparece
    una sola vez en todo el dataset (probablemente artefactos de la
    generación sintética, no carreras reales).

    Parameters
    ----------
    ruta_train_csv:
        Ruta al archivo train.csv de la competencia Kaggle
        "playground-series-s4e6".

    Returns
    -------
    pd.DataFrame
        Datos crudos de Kaggle, sin las filas de ruido.
    """
    datos_crudos = pd.read_csv(ruta_train_csv)
    filtro_sin_ruido = ~datos_crudos["Course"].isin(CODIGOS_CARRERA_RUIDO)
    return datos_crudos[filtro_sin_ruido].copy()


def construir_variables_categoricas(
    datos_crudos: pd.DataFrame, dataset: pd.DataFrame
) -> None:
    """Agrega al ``dataset`` las variables categóricas traducidas/agrupadas.

    Modifica ``dataset`` in-place agregando una columna por cada variable
    categórica de la cátedra, mapeada desde su columna cruda equivalente.
    """
    dataset["Estudios_máximos_antes_de_la_inscripción"] = datos_crudos[
        "Previous qualification"
    ].map(MAPA_ESTUDIOS_MAXIMOS)
    dataset["estado_civil"] = datos_crudos["Marital status"].map(
        MAPA_ESTADO_CIVIL
    )
    dataset["modo_aplicacion"] = datos_crudos["Application mode"].map(
        MAPA_MODO_APLICACION
    )
    dataset["sexo"] = datos_crudos["Gender"].map(MAPA_SEXO)
    dataset["desplazado"] = datos_crudos["Displaced"].map(MAPA_SI_NO)
    dataset["target"] = datos_crudos["Target"].map(MAPA_TARGET)
    dataset["Carrera"] = datos_crudos["Course"].map(MAPA_CARRERA)
    dataset["Asistencia_diurna/nocturna"] = datos_crudos[
        "Daytime/evening attendance"
    ].map(MAPA_TURNO)
    dataset["Necesidades_educativas_especiales"] = datos_crudos[
        "Educational special needs"
    ].map(MAPA_SI_NO)
    dataset["Tiene_deuda"] = datos_crudos["Debtor"].map(MAPA_SI_NO)
    dataset["Pago_de_matrícula_al_día"] = datos_crudos[
        "Tuition fees up to date"
    ].map(MAPA_SI_NO)
    dataset["Poseedor_de_Beca"] = datos_crudos["Scholarship holder"].map(
        MAPA_SI_NO
    )
    dataset["Estudiante_Internacional"] = datos_crudos["International"].map(
        MAPA_SI_NO
    )


def construir_variables_numericas(
    datos_crudos: pd.DataFrame, dataset: pd.DataFrame
) -> None:
    """Agrega al ``dataset`` las variables numéricas, con sus reescalados.

    Las notas de ingreso y de estudios previos vienen en la escala
    portuguesa 0-200, y se llevan a una escala 0-100 dividiendo por 2. Las
    notas de cursada vienen en escala 0-20 y se llevan a 0-100
    multiplicando por 5. El resto de las columnas numéricas (edad,
    cantidades de materias/evaluaciones) se copian directamente porque ya
    están en la escala correcta.

    Modifica ``dataset`` in-place.
    """
    dataset["Cualificación_promedio_de_estudios_previos"] = (
        datos_crudos["Previous qualification (grade)"] / 2
    ).round().astype(int)
    dataset["Puntaje_en_examen_de_ingreso"] = (
        datos_crudos["Admission grade"] / 2
    ).round().astype(int)
    dataset["Edad_al_momento_de_la_inscripción"] = datos_crudos[
        "Age at enrollment"
    ].astype(int)
    dataset["Unidades_curriculares_1er_semestre_(inscrito)"] = datos_crudos[
        "Curricular units 1st sem (enrolled)"
    ].astype(int)
    dataset["Cantidad_de_evaluaciones_en_el_1er_semestre"] = datos_crudos[
        "Curricular units 1st sem (evaluations)"
    ].astype(int)
    dataset["Unidades_curriculares_1er_semestre_(aprobadas)"] = datos_crudos[
        "Curricular units 1st sem (approved)"
    ].astype(int)
    dataset["Nota_promedio_en_el_1er_semestre"] = (
        datos_crudos["Curricular units 1st sem (grade)"] * 5
    ).round().astype(int)
    dataset["Unidades_curriculares_2do_semestre_(inscrito)"] = datos_crudos[
        "Curricular units 2nd sem (enrolled)"
    ].astype(int)
    dataset["Cantidad_de_evaluaciones_en_el_2do_semestre_"] = datos_crudos[
        "Curricular units 2nd sem (evaluations)"
    ].astype(int)
    dataset["Unidades_curriculares_2do_semestre_(aprobadas)"] = datos_crudos[
        "Curricular units 2nd sem (approved)"
    ].astype(int)
    dataset["Nota_promedio_en_el_2do_semestre"] = (
        datos_crudos["Curricular units 2nd sem (grade)"] * 5
    ).round().astype(int)


def construir_variables_padres(
    datos_crudos: pd.DataFrame, dataset: pd.DataFrame
) -> None:
    """Agrega al ``dataset`` la demanda cognitiva y los ingresos de los padres.

    Ambas variables se derivan de la ocupación de cada progenitor por
    separado (no se duplica el valor del padre en la madre), agrupando por
    grupo ocupacional ISCO-08 según ``obtener_grupo_isco``.

    Modifica ``dataset`` in-place.
    """
    grupo_isco_padre = datos_crudos["Father's occupation"].map(
        obtener_grupo_isco
    )
    grupo_isco_madre = datos_crudos["Mother's occupation"].map(
        obtener_grupo_isco
    )

    dataset["Demanda_Cog_Padre"] = grupo_isco_padre.map(
        MAPA_DEMANDA_COGNITIVA_POR_GRUPO_ISCO
    )
    dataset["Demanda_Cog_Madre"] = grupo_isco_madre.map(
        MAPA_DEMANDA_COGNITIVA_POR_GRUPO_ISCO
    )
    dataset["Ingresos_Padre"] = grupo_isco_padre.map(
        MAPA_INGRESOS_POR_GRUPO_ISCO
    )
    dataset["Ingresos_Madre"] = grupo_isco_madre.map(
        MAPA_INGRESOS_POR_GRUPO_ISCO
    )


def construir_dataset_estilo_catedra(
    datos_crudos: pd.DataFrame,
) -> pd.DataFrame:
    """Traduce el dataset crudo de Kaggle al esquema y estilo de la cátedra.

    Parameters
    ----------
    datos_crudos:
        DataFrame con las columnas originales de Kaggle (ya sin las filas
        de ruido), tal como lo devuelve ``cargar_datos_crudos``.

    Returns
    -------
    pd.DataFrame
        Dataset con las 27 variables de la cátedra, en español, con el
        mismo orden de columnas que usa el archivo de la materia.
    """
    dataset = pd.DataFrame(index=datos_crudos.index)

    construir_variables_categoricas(datos_crudos, dataset)
    construir_variables_numericas(datos_crudos, dataset)
    construir_variables_padres(datos_crudos, dataset)

    filas_con_nulos = dataset.isna().any(axis=1).sum()
    if filas_con_nulos:
        columnas_con_nulos = dataset.columns[dataset.isna().any()].tolist()
        raise ValueError(
            f"Quedaron {filas_con_nulos} filas con valores nulos tras el "
            f"mapeo, en las columnas {columnas_con_nulos}. Probablemente "
            "falta agregar algún código crudo a uno de los diccionarios "
            "de mapeo."
        )

    return dataset[ORDEN_COLUMNAS_FINAL]


def submuestrear_estratificado(
    dataset: pd.DataFrame,
    columnas_estrato: List[str],
    filas_objetivo: int,
    semilla: int,
) -> pd.DataFrame:
    """Sub-muestrea ``dataset`` a ``filas_objetivo`` filas, de forma
    estratificada.

    El muestreo estratificado toma, dentro de cada combinación de valores
    de ``columnas_estrato``, la misma proporción de filas. De esta forma se
    preservan tanto las proporciones de esas columnas como -al ser muestreo
    aleatorio simple dentro de cada estrato- la distribución del resto de
    las variables numéricas y categóricas.

    Parameters
    ----------
    dataset:
        Dataset completo del que se quiere tomar una muestra.
    columnas_estrato:
        Columnas que definen los estratos (ej. ["target", "Carrera"]).
    filas_objetivo:
        Cantidad exacta de filas que debe tener la muestra final.
    semilla:
        Semilla del generador aleatorio, para que el resultado sea
        reproducible.

    Returns
    -------
    pd.DataFrame
        Muestra de ``filas_objetivo`` filas, con el índice reiniciado y el
        orden de las filas mezclado al azar.
    """
    generador_aleatorio = np.random.RandomState(semilla)
    proporcion_muestreo = filas_objetivo / len(dataset)

    indices_muestreados: List[int] = []
    grupos = dataset.groupby(columnas_estrato).groups
    for indices_del_grupo in grupos.values():
        indices_del_grupo = np.array(indices_del_grupo)
        cantidad_a_tomar = min(
            round(len(indices_del_grupo) * proporcion_muestreo),
            len(indices_del_grupo),
        )
        indices_muestreados.extend(
            generador_aleatorio.choice(
                indices_del_grupo, size=cantidad_a_tomar, replace=False
            ).tolist()
        )

    muestra = dataset.loc[pd.Index(indices_muestreados)].copy()

    # El redondeo por estrato puede dejar la muestra unas pocas filas por
    # arriba o por debajo del objetivo exacto; se ajusta acá al final.
    diferencia = filas_objetivo - len(muestra)
    indices_restantes = dataset.index.difference(muestra.index)
    if diferencia > 0:
        indices_extra = generador_aleatorio.choice(
            np.array(indices_restantes), size=diferencia, replace=False
        )
        muestra = pd.concat([muestra, dataset.loc[indices_extra]])
    elif diferencia < 0:
        indices_a_quitar = generador_aleatorio.choice(
            np.array(muestra.index), size=-diferencia, replace=False
        )
        muestra = muestra.drop(indices_a_quitar)

    orden_mezclado = generador_aleatorio.permutation(muestra.index.to_numpy())
    return muestra.loc[orden_mezclado].reset_index(drop=True)


def validar_dataset(dataset: pd.DataFrame) -> None:
    """Corre chequeos básicos de calidad sobre el dataset final.

    No devuelve nada: si algún chequeo falla, levanta una excepción para
    frenar la ejecución antes de guardar un dataset con problemas.
    """
    assert dataset.isna().sum().sum() == 0, (
        "El dataset final tiene valores nulos."
    )
    valores_target_esperados = {"Desertor", "En Curso", "Graduado"}
    assert set(dataset["target"].unique()) == valores_target_esperados, (
        "Los valores de 'target' no son los esperados."
    )

    # La demanda cognitiva y los ingresos de padre y madre deben poder
    # diferir entre sí (si coincidieran siempre, indicaría que se están
    # derivando de la misma columna por error).
    coincidencia_demanda = (
        dataset["Demanda_Cog_Padre"] == dataset["Demanda_Cog_Madre"]
    ).mean()
    assert coincidencia_demanda < 0.9, (
        "Demanda_Cog_Padre y Demanda_Cog_Madre coinciden casi siempre: "
        "revisar que se estén derivando de columnas distintas."
    )

    print(
        f"Validación OK. Filas: {len(dataset)}, "
        f"columnas: {dataset.shape[1]}"
    )
    print("Distribución de 'target':")
    print((dataset["target"].value_counts(normalize=True) * 100).round(2))


# --------------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------------- #

def parsear_argumentos() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos del script."""
    parser = argparse.ArgumentParser(
        description=(
            "Genera el dataset final del TP1 a partir del train.csv de la "
            "competencia de Kaggle playground-series-s4e6."
        )
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        required=True,
        help="Ruta al train.csv descargado de Kaggle.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("dataset_TP1_2026.csv"),
        help="Ruta donde guardar el dataset final (default: %(default)s).",
    )
    parser.add_argument(
        "--filas",
        type=int,
        default=FILAS_OBJETIVO_POR_DEFECTO,
        help="Cantidad de filas de la muestra final (default: %(default)s).",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=SEMILLA_POR_DEFECTO,
        help=(
            "Semilla aleatoria, para reproducibilidad "
            "(default: %(default)s)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Orquesta todo el proceso: cargar, transformar, muestrear y guardar."""
    argumentos = parsear_argumentos()

    datos_crudos = cargar_datos_crudos(argumentos.entrada)
    dataset_completo = construir_dataset_estilo_catedra(datos_crudos)

    dataset_final = submuestrear_estratificado(
        dataset_completo,
        columnas_estrato=["target", "Carrera"],
        filas_objetivo=argumentos.filas,
        semilla=argumentos.semilla,
    )

    validar_dataset(dataset_final)

    # utf-8-sig (UTF-8 con BOM) para que Excel muestre bien tildes y eñes
    # al abrir el archivo directamente, sin tener que importarlo a mano.
    dataset_final.to_csv(argumentos.salida, index=False, encoding="utf-8-sig")
    print(f"Dataset final guardado en: {argumentos.salida}")


if __name__ == "__main__":
    main()
