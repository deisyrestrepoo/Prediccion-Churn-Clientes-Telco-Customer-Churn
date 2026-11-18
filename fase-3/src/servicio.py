"""Lógica de servicio de la API: cargar la versión configurada y predecir un registro."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import ARTIFACT_NAME, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, LABEL_NAMES
from .data import preparar_features
from .errors import DatosInvalidosError, ErrorAplicacion, ModeloNoEncontradoError
from .persistence import cargar_modelo, ruta_version


class ModeloNoDisponibleError(ErrorAplicacion):
    """La versión configurada no existe, no se puede cargar o no es compatible (HTTP 503)."""


def versiones_disponibles(models_dir) -> list[str]:
    """Versiones guardadas en models_dir (carpetas que contienen modelo.joblib)."""
    base = Path(models_dir)
    if not base.is_dir():
        return []
    return sorted(c.name for c in base.iterdir() if (c / ARTIFACT_NAME).is_file())


@lru_cache(maxsize=8)
def _cargar_en_cache(ruta: str, mtime_ns: int) -> dict:
    # mtime_ns forma parte de la clave: si el archivo cambia, se vuelve a cargar.
    return cargar_modelo(ruta)


def cargar_modelo_configurado(models_dir, version: str) -> dict:
    """Carga (con caché) la versión indicada o lanza ModeloNoDisponibleError con una guía."""
    disponibles = versiones_disponibles(models_dir)
    ayuda = (f"Versiones disponibles en '{models_dir}': {', '.join(disponibles) or 'ninguna'}. "
             "Configure la variable de entorno MODEL_VERSION con una de ellas y reinicie la API.")
    try:
        archivo = ruta_version(models_dir, version) / ARTIFACT_NAME
    except DatosInvalidosError as exc:
        raise ModeloNoDisponibleError(f"MODEL_VERSION='{version}' no es un identificador válido. {ayuda}") from exc
    if not archivo.is_file():
        raise ModeloNoDisponibleError(f"El modelo de la versión '{version}' no está disponible. {ayuda}")
    try:
        paquete = _cargar_en_cache(str(archivo), archivo.stat().st_mtime_ns)
    except ModeloNoEncontradoError as exc:
        raise ModeloNoDisponibleError(f"No se pudo cargar la versión '{version}': {exc}") from exc
    if list(paquete["esquema"]["variables_entrada"]) != FEATURE_COLUMNS:
        raise ModeloNoDisponibleError(
            f"La versión '{version}' se entrenó con otras variables y no es compatible con esta API. {ayuda}"
        )
    return paquete


def _avisos_categorias_nuevas(pipeline, X: pd.DataFrame) -> list[str]:
    """Categorías que el modelo no vio al entrenar (se predicen igual, sin esa categoría)."""
    try:
        codificador = (pipeline.named_steps["preprocesamiento"]
                       .named_transformers_["categoricas"].named_steps["onehot"])
        conocidas = dict(zip(CATEGORICAL_COLUMNS, codificador.categories_))
    except (KeyError, AttributeError):
        return []
    avisos = []
    for col, cats in conocidas.items():
        nuevas = sorted(set(X[col].dropna().unique()) - set(cats))
        if nuevas:
            avisos.append(f"'{col}' trae categorías no vistas en el entrenamiento {nuevas}; "
                          "se predice igual, sin esa categoría.")
    return avisos


def predecir_registro(paquete: dict, registro: dict) -> dict:
    """Predice un cliente con el mismo preprocesamiento y modelo guardados (sin reajustar nada)."""
    X = preparar_features(pd.DataFrame([registro]))  # una fila, columnas en el orden del esquema
    pipeline = paquete["pipeline"]
    avisos = _avisos_categorias_nuevas(pipeline, X)
    clase = int(pipeline.predict(X)[0])
    probabilidad = float(pipeline.predict_proba(X)[0, 1])
    etiquetas = paquete.get("etiquetas", LABEL_NAMES)
    return {
        "prediccion": etiquetas[clase],
        "clase": clase,
        "probabilidad_churn": round(probabilidad, 4),
        "version": paquete["version"],
        "avisos": avisos,
    }