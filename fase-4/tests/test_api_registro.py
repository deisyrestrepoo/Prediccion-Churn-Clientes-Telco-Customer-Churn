"""Pruebas del registro de solicitudes de la API (Fase 4).

Usan un modelo pequeño entrenado al vuelo con datos sintéticos (el mismo esquema real), así que no
necesitan el CSV de Kaggle ni las versiones de `models/`. El registro se escribe en una carpeta
temporal (ver conftest.py). Se ejecutan desde fase-4/:    python -m pytest -v
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
import monitoreo
from src.data import cargar_datos_etiquetados
from src.persistence import guardar_version
from src.schemas import EJEMPLO_CLIENTE
from src.training import entrenar_y_evaluar
from tests.test_api import _datos_sinteticos

VERSION_PRUEBA = "vtest"
ENCABEZADO = ["fecha_hora", "estado_http", "version", "prediccion"]


@pytest.fixture(scope="module")
def entorno_modelo(tmp_path_factory):
    """Entrena UNA vez un modelo pequeño y lo guarda en una carpeta temporal."""
    base = tmp_path_factory.mktemp("api_registro")
    csv_datos = base / "datos.csv"
    _datos_sinteticos().to_csv(csv_datos, index=False)
    X, y = cargar_datos_etiquetados(csv_datos, "Churn")
    resultado = entrenar_y_evaluar(X, y, seed=42, test_size=0.2, cv_folds=0)
    models_dir = base / "models"
    guardar_version(models_dir, VERSION_PRUEBA, "Churn", 42, csv_datos, len(X), resultado)
    return models_dir


@pytest.fixture
def cliente(entorno_modelo, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(entorno_modelo))
    monkeypatch.setenv("MODEL_VERSION", VERSION_PRUEBA)
    return TestClient(api.app)


def _filas(ruta: Path) -> list[dict]:
    with ruta.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_dos_validas_y_una_sin_campo_dejan_3_filas_y_33_3_por_ciento(cliente, registro_temporal, capsys):
    """El ensayo de la especificación, sobre un registro vacío: una fila por intento, sin duplicados."""
    assert not registro_temporal.exists()                       # registro vacío al empezar
    r1 = cliente.post("/predict", json=EJEMPLO_CLIENTE)
    r2 = cliente.post("/predict", json={**EJEMPLO_CLIENTE, "tenure": 40, "Contract": "Two year"})
    incompleta = {k: v for k, v in EJEMPLO_CLIENTE.items() if k != "Contract"}
    r3 = cliente.post("/predict", json=incompleta)
    assert [r1.status_code, r2.status_code, r3.status_code] == [200, 200, 422]

    filas = _filas(registro_temporal)
    assert len(filas) == 3                                      # si salen 2, falla el manejo de rechazos
    assert [f["estado_http"] for f in filas] == ["200", "200", "422"]
    # En las válidas: la versión y la predicción de la fila son las de la respuesta.
    for fila, respuesta in zip(filas[:2], (r1, r2)):
        assert fila["version"] == respuesta.json()["version"] == VERSION_PRUEBA
        assert fila["prediccion"] == respuesta.json()["prediccion"]
    # En el rechazo: versión y predicción vacías.
    assert filas[2]["version"] == "" and filas[2]["prediccion"] == ""
    assert all(f["fecha_hora"] for f in filas)

    assert monitoreo.main(["--log", str(registro_temporal)]) == 0
    salida = capsys.readouterr().out
    assert "Total de solicitudes: 3" in salida
    assert "Porcentaje de fallos: 33,3 %" in salida


def test_cada_tipo_de_rechazo_deja_una_sola_fila(cliente, registro_temporal):
    """Tipo incompatible, JSON roto y cuerpo vacío: rechazos de validación, una fila cada uno."""
    cliente.post("/predict", json={**EJEMPLO_CLIENTE, "tenure": "abc"})
    cliente.post("/predict", content="{no es json", headers={"content-type": "application/json"})
    cliente.post("/predict", json={})
    assert [f["estado_http"] for f in _filas(registro_temporal)] == ["422", "422", "422"]


def test_modelo_no_disponible_registra_503_sin_version_ni_prediccion(cliente, registro_temporal, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v99")
    assert cliente.post("/predict", json=EJEMPLO_CLIENTE).status_code == 503
    (fila,) = _filas(registro_temporal)
    assert fila["estado_http"] == "503" and fila["version"] == "" and fila["prediccion"] == ""


def test_error_no_controlado_registra_500(entorno_modelo, registro_temporal, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(entorno_modelo))
    monkeypatch.setenv("MODEL_VERSION", VERSION_PRUEBA)

    def falla(*args, **kwargs):
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(api, "predecir_registro", falla)
    c = TestClient(api.app, raise_server_exceptions=False)
    assert c.post("/predict", json=EJEMPLO_CLIENTE).status_code == 500
    (fila,) = _filas(registro_temporal)
    assert fila["estado_http"] == "500" and fila["version"] == "" and fila["prediccion"] == ""


def test_solo_se_registra_predict_y_el_historial_se_conserva(cliente, registro_temporal):
    # /health no genera filas
    assert cliente.get("/health").status_code == 200
    assert not registro_temporal.exists()

    cliente.post("/predict", json=EJEMPLO_CLIENTE)
    # Otra "ejecución": un cliente nuevo sigue agregando al mismo archivo, sin borrar ni repetir encabezado.
    otro = TestClient(api.app)
    otro.post("/predict", json=EJEMPLO_CLIENTE)
    lineas = registro_temporal.read_text(encoding="utf-8").splitlines()
    assert lineas[0].split(",") == ENCABEZADO
    assert len(lineas) == 3 and sum(1 for l in lineas if l.startswith("fecha_hora")) == 1


def test_el_cambio_de_version_se_ve_en_el_registro(cliente, registro_temporal, monkeypatch):
    """La columna `version` permite saber qué modelo predijo cada solicitud."""
    cliente.post("/predict", json=EJEMPLO_CLIENTE)
    base = Path(api.obtener_config().models_dir)
    # Una segunda versión idéntica (copia de la primera) para poder cambiar la configuración.
    import shutil
    shutil.copytree(base / VERSION_PRUEBA, base / "vtest2")
    paquete = base / "vtest2" / "modelo.joblib"
    import joblib
    contenido = joblib.load(paquete)
    contenido["version"] = "vtest2"
    joblib.dump(contenido, paquete)
    monkeypatch.setenv("MODEL_VERSION", "vtest2")
    cliente.post("/predict", json=EJEMPLO_CLIENTE)
    assert [f["version"] for f in _filas(registro_temporal)] == [VERSION_PRUEBA, "vtest2"]
    shutil.rmtree(base / "vtest2")


def test_metodo_no_permitido_en_predict_cuenta_como_intento_fallido(cliente, registro_temporal):
    """Decisión documentada en el README: GET /predict (405) también deja una fila."""
    assert cliente.get("/predict").status_code == 405
    (fila,) = _filas(registro_temporal)
    assert fila["estado_http"] == "405" and fila["version"] == "" and fila["prediccion"] == ""
