"""Pruebas del registro y del script de monitoreo (Fase 4).

Ejecutar desde fase-4:
    python -m pytest

Solo usan archivos temporales (tmp_path): no necesitan la API, el modelo ni la red.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

FASE4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FASE4))

import monitoreo  # noqa: E402
import registro  # noqa: E402

ENCABEZADO = "fecha_hora,estado_http,version,prediccion"


def _escenario_tipico(ruta: Path) -> None:
    """Dos solicitudes válidas y una rechazada por falta de un campo (como pide la Fase 4)."""
    registro.registrar_evento(200, "v1", "No churn", ruta, datetime(2026, 11, 17, 10, 0, 0))
    registro.registrar_evento(200, "v1", "Churn", ruta, datetime(2026, 11, 17, 10, 1, 0))
    registro.registrar_evento(422, None, None, ruta, datetime(2026, 11, 17, 10, 2, 0))


def test_registro_escribe_encabezado_una_vez_y_conserva_el_historial(tmp_path):
    ruta = tmp_path / "logs" / "predicciones.csv"      # la carpeta se crea sola
    registro.registrar_evento(200, "v1", "Churn", ruta, datetime(2026, 11, 17, 10, 0, 0))
    registro.registrar_evento(422, None, None, ruta, datetime(2026, 11, 17, 10, 1, 0))
    # Segunda "ejecución": se vuelve a abrir el archivo y se agrega sin borrar lo anterior.
    registro.registrar_evento(503, None, None, ruta, datetime(2026, 11, 17, 10, 2, 0))

    assert ruta.read_text(encoding="utf-8").splitlines() == [
        ENCABEZADO,
        "2026-11-17T10:00:00,200,v1,Churn",
        "2026-11-17T10:01:00,422,,",           # error: versión y predicción vacías
        "2026-11-17T10:02:00,503,,",
    ]


def test_dos_validas_y_una_invalida_dan_3_solicitudes_y_33_3_por_ciento(tmp_path, capsys):
    ruta = tmp_path / "predicciones.csv"
    _escenario_tipico(ruta)

    indicadores = monitoreo.calcular_indicadores(monitoreo.leer_registro(ruta))
    assert indicadores["total"] == 3
    assert indicadores["fallidas"] == 1
    assert indicadores["porcentaje_fallos"] == pytest.approx(100 / 3)

    assert monitoreo.main(["--log", str(ruta)]) == 0
    salida = capsys.readouterr().out
    assert "Total de solicitudes: 3" in salida
    assert "Porcentaje de fallos: 33,3 %" in salida


def test_cuentan_como_fallidas_solo_4xx_y_5xx(tmp_path):
    ruta = tmp_path / "predicciones.csv"
    for codigo in (200, 201, 399, 400, 422, 503, 599):
        registro.registrar_evento(codigo, None, None, ruta)
    indicadores = monitoreo.calcular_indicadores(monitoreo.leer_registro(ruta))
    assert indicadores["total"] == 7
    assert indicadores["fallidas"] == 4          # 400, 422, 503 y 599


@pytest.mark.parametrize("contenido", ["", ENCABEZADO + "\n"], ids=["archivo vacío", "solo encabezado"])
def test_registro_vacio_muestra_total_0_y_sin_solicitudes(tmp_path, capsys, contenido):
    ruta = tmp_path / "predicciones.csv"
    ruta.write_text(contenido, encoding="utf-8")

    assert monitoreo.main(["--log", str(ruta)]) == 0
    salida = capsys.readouterr().out
    assert "Total de solicitudes: 0" in salida
    assert "Porcentaje de fallos: sin solicitudes" in salida


def test_registro_inexistente_o_invalido_da_error_claro_y_codigo_1(tmp_path, capsys):
    # Archivo que no existe
    assert monitoreo.main(["--log", str(tmp_path / "no_existe.csv")]) == 1
    assert "No se encontró el archivo de registro" in capsys.readouterr().err

    # Encabezado distinto del esperado
    ajeno = tmp_path / "ajeno.csv"
    ajeno.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    assert monitoreo.main(["--log", str(ajeno)]) == 1
    assert "encabezado esperado" in capsys.readouterr().err

    # estado_http que no es un número
    malo = tmp_path / "malo.csv"
    malo.write_text(ENCABEZADO + "\n2026-11-17T10:00:00,abc,v1,Churn\n", encoding="utf-8")
    assert monitoreo.main(["--log", str(malo)]) == 1
    assert "no es un número entero" in capsys.readouterr().err


def test_script_como_proceso_usa_la_variable_PREDICTION_LOG(tmp_path):
    ruta = tmp_path / "predicciones.csv"
    _escenario_tipico(ruta)
    proceso = subprocess.run(
        [sys.executable, str(FASE4 / "monitoreo.py")],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PREDICTION_LOG": str(ruta), "PYTHONUTF8": "1"},
    )
    assert proceso.returncode == 0, proceso.stderr
    assert "Total de solicitudes: 3" in proceso.stdout
    assert "33,3 %" in proceso.stdout
