"""API REST de predicción de churn (Fase 3).

Arranque (desde fase-3/):
    uvicorn api:app --host 0.0.0.0 --port 8000

Documentación interactiva: http://localhost:8000/docs
La versión del modelo se elige con la variable MODEL_VERSION (ver src/settings.py).
"""
from __future__ import annotations

from fastapi import FastAPI

from src.settings import modelo_disponible, obtener_config

app = FastAPI(
    title="API de predicción de churn",
    description="Consulta un modelo guardado y entrena nuevas versiones mediante HTTP "
                "(Proyecto Integrador, Modelos y Simulación de Sistemas I).",
    version="0.1.0",
)


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