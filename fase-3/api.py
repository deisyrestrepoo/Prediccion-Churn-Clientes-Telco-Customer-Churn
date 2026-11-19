"""API REST de predicción de churn (Fase 3).

Arranque (desde fase-3/):
    uvicorn api:app --host 0.0.0.0 --port 8000

Documentación interactiva: http://localhost:8000/docs
La versión del modelo se elige con la variable MODEL_VERSION (ver src/settings.py).
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.errors import DatosInvalidosError, ErrorAplicacion
from src.schemas import ClienteEntrada, EntrenamientoEntrada, EntrenamientoSalida, PrediccionSalida
from src.servicio import (
    EntrenamientoEnCursoError,
    ModeloNoDisponibleError,
    cargar_modelo_configurado,
    entrenar_nueva_version,
    predecir_registro,
)
from src.settings import modelo_disponible, obtener_config, obtener_config_entrenamiento

app = FastAPI(
    title="API de predicción de churn",
    description="Consulta un modelo guardado y entrena nuevas versiones mediante HTTP "
                "(Proyecto Integrador, Modelos y Simulación de Sistemas I).",
    version="0.3.0",
)


# ----------------------------------------------------------------------------- errores
def respuesta_error(status_code: int, error: str, mensaje: str, detalles: list | None = None) -> JSONResponse:
    """Formato único de error: {"error", "mensaje", "detalles"?}."""
    cuerpo = {"error": error, "mensaje": mensaje}
    if detalles:
        cuerpo["detalles"] = detalles
    return JSONResponse(status_code=status_code, content=cuerpo)


def _problema(err: dict) -> tuple[str, str]:
    """Traduce un error de validación de pydantic a (campo, problema) en español."""
    loc, tipo, ctx = err["loc"], err["type"], err.get("ctx") or {}
    campo = loc[1] if len(loc) >= 2 and isinstance(loc[1], str) else "cuerpo"
    if tipo == "missing":
        return campo, "campo obligatorio ausente"
    if tipo in ("int_type", "int_parsing", "int_from_float") and len(loc) <= 2:
        return campo, "debe ser un número entero (sin comillas)"
    if tipo in ("float_type", "float_parsing", "int_type", "int_parsing"):
        return campo, "debe ser un número (sin comillas)"
    if tipo == "string_type":
        return campo, "debe ser texto"
    if tipo == "literal_error":
        return campo, "valor no permitido (se esperaba " + str(ctx.get("expected", "")).replace(" or ", " o ") + ")"
    if tipo in ("greater_than_equal", "greater_than"):
        return campo, f"debe ser mayor o igual a {ctx.get('ge', 0)}"
    if tipo == "json_invalid":
        return "cuerpo", "no es un JSON válido"
    if tipo in ("model_attributes_type", "dict_type", "model_type"):
        return "cuerpo", "debe ser un objeto JSON con las variables del cliente"
    return campo, err.get("msg", "valor inválido")


@app.exception_handler(RequestValidationError)
async def manejar_validacion(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Entrada inválida -> HTTP 422 con el nombre de cada campo problemático.

    Un solo punto atiende todos los rechazos de validación (útil para registrarlos en la Fase 4).
    """
    detalles, vistos = [], set()
    for err in exc.errors():
        campo, problema = _problema(err)
        if campo not in vistos:  # un solo problema por campo
            vistos.add(campo)
            detalles.append({"campo": campo, "problema": problema})
    mensaje = "; ".join(f"{d['campo']}: {d['problema']}" for d in detalles)
    return respuesta_error(422, "Entrada inválida", mensaje, detalles)


@app.exception_handler(ErrorAplicacion)
async def manejar_error_aplicacion(request: Request, exc: ErrorAplicacion) -> JSONResponse:
    """Errores previsibles no atendidos en la ruta (p. ej. una variable de entorno inválida)."""
    return respuesta_error(500, "Error de configuración o de la aplicación", str(exc))


# ------------------------------------------------------------------------------ rutas
@app.get("/health", summary="Estado del servicio")
def health() -> dict:
    """Indica que la API está en ejecución y qué versión del modelo tiene configurada.

    Responde 200 aunque el modelo no esté disponible: /health describe el servicio,
    no la predicción (esa falla con 503 en /predict).
    """
    cfg = obtener_config()
    return {
        "estado": "ok",
        "mensaje": "API en ejecución",
        "version_configurada": cfg.version,
        "modelo_disponible": modelo_disponible(cfg),
    }


@app.post(
    "/predict",
    response_model=PrediccionSalida,
    summary="Predice el churn de un cliente",
    responses={
        422: {"description": "Falta una variable obligatoria o un valor tiene un tipo incompatible."},
        503: {"description": "La versión configurada del modelo no está disponible."},
    },
)
def predict(cliente: ClienteEntrada):
    """Recibe las 19 variables predictoras de un cliente (no necesita `Churn`) y devuelve la
    etiqueta predicha, la probabilidad y la versión del modelo utilizada.

    Un valor `null` se trata como dato faltante con la regla guardada en la Fase 2.
    """
    cfg = obtener_config()
    try:
        paquete = cargar_modelo_configurado(cfg.models_dir, cfg.version)
    except ModeloNoDisponibleError as exc:
        return respuesta_error(503, "Modelo no disponible", str(exc))
    try:
        return predecir_registro(paquete, cliente.model_dump())
    except DatosInvalidosError as exc:  # defensa: la validación ya cubre estos casos
        return respuesta_error(422, "Entrada inválida", str(exc))


@app.post(
    "/train",
    status_code=201,
    response_model=EntrenamientoSalida,
    summary="Entrena y guarda una versión nueva del modelo",
    responses={
        409: {"description": "Ya hay un entrenamiento en curso."},
        503: {"description": "Los datos de entrenamiento no están disponibles o no son válidos."},
    },
)
def train(entrada: Optional[EntrenamientoEntrada] = None):
    """Entrena con los datos y la configuración de la Fase 2 y guarda una versión NUEVA con su
    modelo, métricas y metadatos. Las versiones anteriores no se modifican.

    El cuerpo puede ir vacío. Tarda alrededor de un minuto (incluye validación cruzada).
    La nueva versión no se activa sola: `/predict` sigue usando `MODEL_VERSION` hasta que se
    cambie esa variable y se reinicie la API.
    """
    cfg = obtener_config()
    entrenamiento = obtener_config_entrenamiento()
    seed = entrada.seed if entrada is not None and entrada.seed is not None else entrenamiento.seed
    try:
        resultado = entrenar_nueva_version(cfg.models_dir, cfg.data_path, cfg.target, seed,
                                           entrenamiento.test_size, entrenamiento.cv_folds)
    except EntrenamientoEnCursoError as exc:
        return respuesta_error(409, "Entrenamiento en curso", str(exc))
    except DatosInvalidosError as exc:
        if not cfg.data_path.is_file():
            return respuesta_error(
                503, "Datos de entrenamiento no disponibles",
                f"{exc} Coloque el CSV en '{cfg.data_path.as_posix()}' (o monte la carpeta de datos en Docker) "
                "o configure la variable de entorno DATA_PATH.")
        return respuesta_error(500, "No se pudo entrenar", str(exc))
    resultado["version_activa"] = cfg.version
    resultado["mensaje"] = (
        f"Versión {resultado['version']} guardada; las anteriores se conservan. /predict sigue usando "
        f"'{cfg.version}'. Para activar la nueva, configure MODEL_VERSION={resultado['version']} y reinicie la API.")
    return resultado