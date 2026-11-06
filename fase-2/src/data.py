"""Lectura y validación de datos. Se usa igual en entrenamiento y predicción."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_MAPPING,
)
from .errors import DatosInvalidosError


def leer_csv(ruta) -> pd.DataFrame:
    """Lee un CSV y devuelve errores claros si no existe o no se puede leer."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise DatosInvalidosError(f"No se encontró el archivo CSV: {ruta}")
    try:
        df = pd.read_csv(ruta)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatosInvalidosError(f"No se pudo leer el CSV {ruta}: {exc}") from exc
    if df.empty:
        raise DatosInvalidosError(f"El archivo CSV no tiene filas: {ruta}")
    return df


def validar_columnas(df: pd.DataFrame, requeridas: list[str]) -> None:
    """Verifica que existan las columnas requeridas (una celda nula NO es una columna ausente)."""
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise DatosInvalidosError(
            "Faltan columnas requeridas: " + ", ".join(faltantes)
        )


def _a_numerico(serie: pd.Series, nombre: str) -> pd.Series:
    """Convierte a float. Vacíos/espacios -> NaN (valor faltante legítimo);
    texto no numérico -> error (tipo incompatible)."""
    texto = serie.astype("string").str.strip()
    texto = texto.mask(texto == "", pd.NA)
    numerico = pd.to_numeric(texto, errors="coerce")
    invalidos = numerico.isna() & texto.notna()
    if invalidos.any():
        ejemplos = texto[invalidos].unique()[:3].tolist()
        raise DatosInvalidosError(
            f"La columna '{nombre}' debe ser numérica, pero tiene valores no numéricos "
            f"(ej.: {ejemplos})."
        )
    return numerico.astype("float64")


def _a_texto(serie: pd.Series) -> pd.Series:
    """Convierte a texto limpio; vacíos -> NaN (los imputa el pipeline)."""
    texto = serie.astype("string").str.strip()
    texto = texto.mask(texto == "", pd.NA)
    return texto.astype(object).where(texto.notna(), np.nan)


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deja las variables predictoras con el orden y los tipos del entrenamiento.

    - Ignora columnas adicionales (customerID, Churn, etc.) y el orden del CSV.
    - No imputa ni escala: eso lo hace el Pipeline guardado con el modelo.
    """
    validar_columnas(df, FEATURE_COLUMNS)
    X = pd.DataFrame(index=df.index)
    for col in FEATURE_COLUMNS:
        if col in NUMERIC_COLUMNS or col in BINARY_NUMERIC_COLUMNS:
            X[col] = _a_numerico(df[col], col)
        else:
            X[col] = _a_texto(df[col])
    return X


def cargar_datos_etiquetados(ruta, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Lee el CSV de entrenamiento y devuelve (X, y) con y en {0, 1}."""
    df = leer_csv(ruta)
    validar_columnas(df, FEATURE_COLUMNS + [target])

    etiquetas = df[target].astype("string").str.strip()
    y = etiquetas.map(TARGET_MAPPING)
    if y.isna().any():
        invalidas = sorted(etiquetas[y.isna()].dropna().unique().tolist())
        raise DatosInvalidosError(
            f"La variable objetivo '{target}' solo admite los valores "
            f"{list(TARGET_MAPPING)}; se encontraron nulos o valores no válidos: {invalidas}"
        )
    return preparar_features(df), y.astype(int)