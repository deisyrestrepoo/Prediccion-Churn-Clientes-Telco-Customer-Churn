"""Registro persistente de las solicitudes a POST /predict (Fase 4).

Cada intento de predicción (aceptado o rechazado) agrega UNA fila a un archivo CSV:

    fecha_hora,estado_http,version,prediccion
    2026-11-17T10:00:00,200,v1,Churn
    2026-11-17T10:02:00,422,,

- El archivo se abre en modo "agregar": el historial se conserva entre ejecuciones.
- El encabezado se escribe solo cuando el archivo se crea (o está vacío).
- En un error, `prediccion` queda vacía y, si no se usó ningún modelo, `version` también.
- Ubicación: variable de entorno PREDICTION_LOG (por defecto logs/predicciones.csv).
  En Docker debe apuntar a una carpeta montada para que el registro persista.

Este módulo no depende de FastAPI: la API solo llama a `registrar_evento`.
"""
from __future__ import annotations

import csv
import os
import threading
from datetime import datetime
from pathlib import Path

CAMPOS = ["fecha_hora", "estado_http", "version", "prediccion"]
LOG_POR_DEFECTO = "logs/predicciones.csv"

# Las rutas sincrónicas de FastAPI se ejecutan en varios hilos: el candado evita que dos
# solicitudes escriban el encabezado o mezclen filas al mismo tiempo.
_candado = threading.Lock()


def ruta_registro() -> Path:
    """Ruta del registro según la variable PREDICTION_LOG (se lee en cada llamada)."""
    return Path(os.environ.get("PREDICTION_LOG", LOG_POR_DEFECTO))


def registrar_evento(estado_http: int, version: str | None = None,
                     prediccion: str | None = None, ruta=None,
                     ahora: datetime | None = None) -> None:
    """Agrega una fila al registro. `ruta` y `ahora` existen para poder probarlo."""
    destino = Path(ruta) if ruta is not None else ruta_registro()
    fecha_hora = (ahora or datetime.now()).isoformat(timespec="seconds")
    fila = [
        fecha_hora,
        int(estado_http),
        "" if version is None else version,
        "" if prediccion is None else prediccion,
    ]
    with _candado:
        destino.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not destino.exists() or destino.stat().st_size == 0
        with destino.open("a", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo, lineterminator="\n")
            if nuevo:
                escritor.writerow(CAMPOS)
            escritor.writerow(fila)
