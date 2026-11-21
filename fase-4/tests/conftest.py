"""Configuración común de las pruebas de fase-4.

Ninguna prueba debe escribir en el registro real (logs/predicciones.csv): este fixture, que se
aplica solo, redirige PREDICTION_LOG a una carpeta temporal en cada prueba. Las pruebas que
revisan el registro leen esa misma ruta.
"""
import pytest


@pytest.fixture(autouse=True)
def registro_temporal(tmp_path, monkeypatch):
    ruta = tmp_path / "registro" / "predicciones.csv"
    monkeypatch.setenv("PREDICTION_LOG", str(ruta))
    return ruta
