"""Lógica de servicio de la API: cargar la versión configurada y predecir un registro."""
from __future__ import annotations

import re
import threading
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import ARTIFACT_NAME, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, LABEL_NAMES
from .data import cargar_datos_etiquetados, preparar_features
from .errors import (
    DatosInvalidosError,
    ErrorAplicacion,
    ModeloNoEncontradoError,
    VersionExistenteError,
)
from .persistence import cargar_modelo, guardar_version, ruta_version
from .training import METRICA_PRINCIPAL, entrenar_y_evaluar


class ModeloNoDisponibleError(ErrorAplicacion):
    """La versión configurada no existe, no se puede cargar o no es compatible (HTTP 503)."""


class EntrenamientoEnCursoError(ErrorAplicacion):
    """Ya hay un entrenamiento en ejecución (HTTP 409)."""


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


# ------------------------------------------------------------------ entrenamiento
_BLOQUEO_ENTRENAMIENTO = threading.Lock()
_VERSION_NUMERICA = re.compile(r"^v(\d+)$")
_METRICAS_RESUMEN = ("roc_auc", "recall", "precision", "f1", "accuracy")


def siguiente_version(models_dir) -> str:
    """Primer identificador vN libre: v(mayor número existente + 1). Ignora v-docker, etc."""
    base = Path(models_dir)
    numeros = []
    if base.is_dir():
        for carpeta in base.iterdir():
            coincide = _VERSION_NUMERICA.match(carpeta.name)
            if coincide:
                numeros.append(int(coincide.group(1)))
    return f"v{max(numeros, default=0) + 1}"


def entrenar_nueva_version(models_dir, data_path, target: str, seed: int,
                           test_size: float, cv_folds: int) -> dict:
    """Entrena con la misma lógica de la Fase 2 y guarda una versión NUEVA (nunca sobrescribe).

    Solo se permite un entrenamiento a la vez (si hay otro en curso, EntrenamientoEnCursoError).
    """
    if not _BLOQUEO_ENTRENAMIENTO.acquire(blocking=False):
        raise EntrenamientoEnCursoError(
            "Ya hay un entrenamiento en curso. Espere a que termine y vuelva a intentarlo.")
    try:
        X, y = cargar_datos_etiquetados(data_path, target)
        resultado = entrenar_y_evaluar(X, y, seed, test_size, cv_folds)
        for _ in range(5):  # por si otro proceso reserva el mismo identificador
            version = siguiente_version(models_dir)
            try:
                rutas = guardar_version(models_dir, version, target, seed, data_path, len(X), resultado)
                break
            except VersionExistenteError:
                continue
        else:
            raise ErrorAplicacion("No se pudo reservar un identificador de versión disponible.")
    finally:
        _BLOQUEO_ENTRENAMIENTO.release()

    particion = resultado["particion"]
    return {
        "version": version,
        "metrica_principal": METRICA_PRINCIPAL,
        "seed": seed,
        "observaciones": len(X),
        "filas_entrenamiento": particion["filas_entrenamiento"],
        "filas_prueba": particion["filas_prueba"],
        "metricas": {m: round(resultado["metricas_modelo"][m], 4) for m in _METRICAS_RESUMEN},
        "metricas_modelo_base": {m: round(resultado["metricas_base"][m], 4) for m in _METRICAS_RESUMEN},
        "archivos": {nombre: Path(ruta).as_posix() for nombre, ruta in rutas.items()},
    }