"""Configuración de la API mediante variables de entorno.

Se leen en cada llamada (no al importar el módulo), de modo que cambiar la variable
y reiniciar el servidor basta para elegir otra versión del modelo.

Variables (todas opcionales):
    MODEL_VERSION  Versión que usa /predict (por defecto: v1).
    MODELS_DIR     Carpeta con las versiones guardadas (por defecto: models).
    DATA_PATH      CSV con datos etiquetados para /train.
    TARGET         Variable objetivo para /train (por defecto: Churn).
    SEED           Semilla de /train (por defecto: 42).
    TEST_SIZE      Proporción de prueba de /train (por defecto: 0.2).
    CV_FOLDS       Folds de validación cruzada de /train (por defecto: 5).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import ARTIFACT_NAME, TARGET_DEFAULT
from .errors import ErrorAplicacion
from .persistence import ruta_version

DATA_PATH_DEFAULT = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


@dataclass(frozen=True)
class Config:
    version: str
    models_dir: Path
    data_path: Path
    target: str
    seed: int
    test_size: float
    cv_folds: int


def _leer(nombre: str, defecto: str, tipo):
    valor = os.environ.get(nombre, defecto)
    try:
        return tipo(valor)
    except ValueError as exc:
        raise ErrorAplicacion(
            f"La variable de entorno {nombre}='{valor}' no es válida (se esperaba {tipo.__name__})."
        ) from exc


def obtener_config() -> Config:
    return Config(
        version=os.environ.get("MODEL_VERSION", "v1"),
        models_dir=Path(os.environ.get("MODELS_DIR", "models")),
        data_path=Path(os.environ.get("DATA_PATH", DATA_PATH_DEFAULT)),
        target=os.environ.get("TARGET", TARGET_DEFAULT),
        seed=_leer("SEED", "42", int),
        test_size=_leer("TEST_SIZE", "0.2", float),
        cv_folds=_leer("CV_FOLDS", "5", int),
    )


def modelo_disponible(cfg: Config) -> bool:
    """True si existe el artefacto de la versión configurada (no lo carga)."""
    try:
        return (ruta_version(cfg.models_dir, cfg.version) / ARTIFACT_NAME).is_file()
    except ErrorAplicacion:  # identificador de versión no válido
        return False