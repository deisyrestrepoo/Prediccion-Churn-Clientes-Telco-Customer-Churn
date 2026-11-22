"""Guardado, versionamiento y carga de modelos."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from .config import (
    ARTIFACT_NAME,
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATA_SOURCE,
    FEATURE_COLUMNS,
    LABEL_NAMES,
    METADATA_NAME,
    METRICS_NAME,
    NUMERIC_COLUMNS,
)
from .errors import DatosInvalidosError, ModeloNoEncontradoError, VersionExistenteError
from .training import METRICA_PRINCIPAL

_VERSION_VALIDA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validar_identificador(version: str) -> str:
    """El identificador se usa como nombre de carpeta: evita rutas o caracteres raros."""
    if not _VERSION_VALIDA.match(version or ""):
        raise DatosInvalidosError(
            f"Identificador de versión no válido: '{version}'. "
            "Use letras, números, '.', '_' o '-' (ej.: v1, v2, 2026-10-22)."
        )
    return version


def ruta_version(output_dir, version: str) -> Path:
    return Path(output_dir) / validar_identificador(version)


def verificar_version_libre(output_dir, version: str) -> None:
    """Falla de inmediato si la versión ya existe (nunca se sobrescribe)."""
    destino = ruta_version(output_dir, version)
    if destino.exists():
        raise VersionExistenteError(
            f"La versión '{version}' ya existe en {destino}. "
            "Elija otro identificador (los modelos guardados no se sobrescriben)."
        )


def _sha256(ruta) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def _git(*args) -> str | None:
    try:
        salida = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return salida.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def info_codigo() -> dict:
    """Hash del commit (si hay Git disponible) para identificar el código usado."""
    commit = _git("rev-parse", "HEAD")
    cambios = _git("status", "--porcelain")
    return {
        "commit": commit or "no disponible (sin Git, p. ej. dentro de Docker)",
        "cambios_sin_commit": bool(cambios) if cambios is not None else None,
    }


def _tipos_esperados() -> dict:
    tipos = {c: "numerico" for c in NUMERIC_COLUMNS}
    tipos.update({c: "binario_numerico (0/1)" for c in BINARY_NUMERIC_COLUMNS})
    tipos.update({c: "categorico (texto)" for c in CATEGORICAL_COLUMNS})
    return tipos


def construir_metadata(version, target, seed, ruta_datos, n_obs, resultado) -> dict:
    clasificador = resultado["modelo"].named_steps["clasificador"]
    return {
        "version": version,
        "fecha_creacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo_problema": "clasificacion_binaria",
        "variable_objetivo": target,
        "significado_etiquetas": {str(k): v for k, v in LABEL_NAMES.items()},
        "algoritmo": "LogisticRegression (Pipeline: imputacion + StandardScaler + OneHotEncoder)",
        "parametros_modelo": {
            k: v for k, v in clasificador.get_params().items()
            if k in ("class_weight", "max_iter", "C", "solver", "random_state")
        },
        "semilla": seed,
        "variables_entrada": FEATURE_COLUMNS,
        "tipos_esperados": _tipos_esperados(),
        "datos": {
            "fuente": DATA_SOURCE,
            "archivo": Path(ruta_datos).name,
            "sha256": _sha256(ruta_datos),
            "observaciones": int(n_obs),
        },
        "particion": resultado["particion"],
        "metrica_principal": METRICA_PRINCIPAL,
        "entorno": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "dependencias": "requirements.txt",
        },
        "codigo": info_codigo(),
    }


def guardar_version(output_dir, version, target, seed, ruta_datos, n_obs, resultado) -> dict:
    """Crea models/<version>/ con modelo.joblib, metrics.json y metadata.json.

    La carpeta se crea con exist_ok=False: si ya existe, se rechaza (sin sobrescribir).
    """
    destino = ruta_version(output_dir, version)
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        destino.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise VersionExistenteError(
            f"La versión '{version}' ya existe en {destino}."
        ) from exc

    metadata = construir_metadata(version, target, seed, ruta_datos, n_obs, resultado)

    # El artefacto incluye el Pipeline completo (preprocesamiento + modelo) y el esquema.
    paquete = {
        "pipeline": resultado["modelo"],
        "version": version,
        "variable_objetivo": target,
        "etiquetas": LABEL_NAMES,
        "esquema": {
            "variables_entrada": FEATURE_COLUMNS,
            "numericas": NUMERIC_COLUMNS,
            "binarias_numericas": BINARY_NUMERIC_COLUMNS,
            "categoricas": CATEGORICAL_COLUMNS,
        },
    }
    metricas = {
        "metrica_principal": METRICA_PRINCIPAL,
        "conjunto_evaluacion": "prueba (no usado para ajustar nada)",
        "modelo": resultado["metricas_modelo"],
        "modelo_base": resultado["metricas_base"],
        "validacion_cruzada_entrenamiento": resultado["validacion_cruzada"],
    }

    rutas = {
        "modelo": destino / ARTIFACT_NAME,
        "metricas": destino / METRICS_NAME,
        "metadata": destino / METADATA_NAME,
    }
    joblib.dump(paquete, rutas["modelo"])
    for clave, contenido in (("metricas", metricas), ("metadata", metadata)):
        with open(rutas[clave], "w", encoding="utf-8") as f:
            json.dump(contenido, f, indent=2, ensure_ascii=False)
    return rutas


def cargar_modelo(ruta) -> dict:
    """Carga un artefacto (archivo .joblib o carpeta de la versión). Lo usará predict.py."""
    ruta = Path(ruta)
    if ruta.is_dir():
        ruta = ruta / ARTIFACT_NAME
    if not ruta.is_file():
        raise ModeloNoEncontradoError(f"No se encontró el modelo: {ruta}")
    try:
        paquete = joblib.load(ruta)
    except Exception as exc:  # archivo corrupto o de otra versión de sklearn
        raise ModeloNoEncontradoError(f"No se pudo cargar el modelo {ruta}: {exc}") from exc
    if not isinstance(paquete, dict) or "pipeline" not in paquete or "esquema" not in paquete:
        raise ModeloNoEncontradoError(
            f"El archivo {ruta} no es un artefacto válido de este proyecto."
        )
    return paquete