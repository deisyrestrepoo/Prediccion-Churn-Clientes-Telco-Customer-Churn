"""Indicadores de operación de la API a partir del registro de solicitudes (Fase 4).

Muestra dos números:
  - total de solicitudes a /predict (filas de datos del registro), y
  - porcentaje de solicitudes fallidas (respuestas HTTP 4xx o 5xx).

Si el registro está vacío: total 0 y «sin solicitudes» como porcentaje.
Este porcentaje describe el funcionamiento del servicio, NO la calidad predictiva
(medirla exigiría conocer los resultados reales).

Ejemplo (desde fase-4):
    python monitoreo.py --log logs/predicciones.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from registro import CAMPOS, ruta_registro


class ErrorRegistro(Exception):
    """Problema previsible con el archivo de registro (se muestra sin traceback)."""


def leer_registro(ruta) -> list[dict]:
    """Lee las filas de datos del registro. Un archivo vacío o con solo encabezado da []."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ErrorRegistro(f"No se encontró el archivo de registro: {ruta}")
    with ruta.open(newline="", encoding="utf-8") as archivo:
        lector = csv.reader(archivo)
        encabezado = next(lector, None)
        if encabezado is None:          # archivo de 0 bytes
            return []
        if encabezado != CAMPOS:
            raise ErrorRegistro(
                f"El registro {ruta} no tiene el encabezado esperado: {','.join(CAMPOS)}"
            )
        filas = []
        for numero, fila in enumerate(lector, start=2):
            if not fila:                # línea en blanco
                continue
            if len(fila) != len(CAMPOS):
                raise ErrorRegistro(f"Línea {numero} del registro con {len(fila)} columnas "
                                    f"(se esperaban {len(CAMPOS)}).")
            datos = dict(zip(CAMPOS, fila))
            try:
                datos["estado_http"] = int(datos["estado_http"])
            except ValueError:
                raise ErrorRegistro(
                    f"Línea {numero} del registro: estado_http '{datos['estado_http']}' "
                    "no es un número entero."
                ) from None
            filas.append(datos)
        return filas


def calcular_indicadores(filas: list[dict]) -> dict:
    """Total, fallidas (4xx/5xx) y porcentaje (None si no hay solicitudes)."""
    total = len(filas)
    fallidas = sum(1 for f in filas if 400 <= f["estado_http"] <= 599)
    porcentaje = None if total == 0 else fallidas / total * 100
    return {"total": total, "fallidas": fallidas, "porcentaje_fallos": porcentaje}


def _formatear_porcentaje(valor: float | None) -> str:
    if valor is None:
        return "sin solicitudes"
    return f"{valor:.1f}".replace(".", ",") + " %"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Muestra el total de solicitudes a /predict y el porcentaje de fallos."
    )
    parser.add_argument("--log", default=None,
                        help="Ruta del registro CSV (por defecto: PREDICTION_LOG o "
                             "logs/predicciones.csv).")
    args = parser.parse_args(argv)
    ruta = Path(args.log) if args.log else ruta_registro()
    try:
        indicadores = calcular_indicadores(leer_registro(ruta))
    except ErrorRegistro as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Registro: {ruta}")
    print(f"Total de solicitudes: {indicadores['total']}")
    print(f"Solicitudes fallidas (HTTP 4xx o 5xx): {indicadores['fallidas']}")
    print(f"Porcentaje de fallos: {_formatear_porcentaje(indicadores['porcentaje_fallos'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
