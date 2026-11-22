#!/usr/bin/env python3
"""Cliente de prueba de la API de churn (solo biblioteca estándar).

Toma una fila del CSV de ejemplo del proyecto, la convierte en JSON (números como números,
categorías como texto, celdas vacías como null) y la envía a POST /predict.

Con la API en marcha, desde fase-3/:
    python cliente_prueba.py
    python cliente_prueba.py --fila 6                 # fila 7 del CSV (TotalCharges vacío -> null)
    python cliente_prueba.py --quitar Contract        # debe responder 422 nombrando el campo
    python cliente_prueba.py --url http://127.0.0.1:8001

Código de salida 0 si la API respondió lo esperado (200, o 422 con --quitar); 1 en otro caso.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request

ENTEROS = {"SeniorCitizen", "tenure"}
DECIMALES = {"MonthlyCharges", "TotalCharges"}
IDENTIFICADOR = "customerID"  # no es predictora: se muestra pero no se envía


def leer_fila(ruta_csv: str, fila: int) -> tuple[str, dict]:
    """Devuelve (customerID, registro JSON) de la fila indicada (0 = primera de datos)."""
    with open(ruta_csv, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not 0 <= fila < len(filas):
        raise SystemExit(f"--fila debe estar entre 0 y {len(filas) - 1} (el CSV tiene {len(filas)} filas).")
    cruda = filas[fila]
    registro = {}
    for columna, valor in cruda.items():
        if columna == IDENTIFICADOR:
            continue
        valor = valor.strip()
        if valor == "":
            registro[columna] = None
        elif columna in ENTEROS:
            registro[columna] = int(valor)
        elif columna in DECIMALES:
            registro[columna] = float(valor)
        else:
            registro[columna] = valor
    return cruda.get(IDENTIFICADOR, f"fila {fila}"), registro


def solicitar(url: str, cuerpo: dict | None = None) -> tuple[int, dict]:
    datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    peticion = urllib.request.Request(url, data=datos, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(peticion, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as exc:  # la API respondió con un error (4xx o 5xx)
        texto = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(texto)
        except ValueError:
            return exc.code, {"mensaje": texto.strip() or exc.reason}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Envía un cliente del CSV de ejemplo a POST /predict.")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="Dirección de la API (por defecto: %(default)s).")
    p.add_argument("--csv", default="ejemplos/ejemplo_prediccion.csv", help="CSV de ejemplo (por defecto: %(default)s).")
    p.add_argument("--fila", type=int, default=0, help="Fila de datos a enviar, desde 0 (por defecto: 0).")
    p.add_argument("--quitar", metavar="CAMPO", help="Quita ese campo del JSON para probar la validación.")
    args = p.parse_args(argv)

    cliente_id, registro = leer_fila(args.csv, args.fila)
    if args.quitar:
        if args.quitar not in registro:
            raise SystemExit(f"El campo '{args.quitar}' no existe en el registro.")
        del registro[args.quitar]

    try:
        estado, salud = solicitar(args.url + "/health")
        print(f"GET /health -> HTTP {estado}: {json.dumps(salud, ensure_ascii=False)}")
        print(f"\nPOST /predict con el cliente {cliente_id}"
              + (f" (sin '{args.quitar}')" if args.quitar else "") + ":")
        print(json.dumps(registro, indent=2, ensure_ascii=False))
        codigo, respuesta = solicitar(args.url + "/predict", registro)
    except OSError as exc:  # URLError, conexión rechazada o cortada, tiempo de espera agotado
        motivo = getattr(exc, "reason", None) or exc
        print(f"No se pudo conectar con la API en {args.url}: {motivo}\n"
              "¿Está en ejecución y ya terminó de arrancar? "
              "(uvicorn api:app, o el contenedor Docker: espere unos segundos y revise 'docker logs churn-api')",
              file=sys.stderr)
        return 1

    print(f"\n-> HTTP {codigo}:")
    print(json.dumps(respuesta, indent=2, ensure_ascii=False))
    esperado = 422 if args.quitar else 200
    return 0 if codigo == esperado else 1


if __name__ == "__main__":
    sys.exit(main())