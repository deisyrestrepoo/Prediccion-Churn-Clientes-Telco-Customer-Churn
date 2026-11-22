"""Pruebas de la API (Fase 3).

Usan un modelo pequeño entrenado al vuelo con datos sintéticos que tienen el esquema real:
no necesitan el CSV de Kaggle, ni la red, ni las versiones de `models/`. Todo se escribe en
directorios temporales, así que no modifican los modelos de la entrega.

Se ejecutan desde fase-3/:    python -m pytest -v
Las tres pruebas mínimas de la especificación son las tres primeras.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
from src.data import cargar_datos_etiquetados, preparar_features
from src.persistence import cargar_modelo, guardar_version
from src.schemas import EJEMPLO_CLIENTE
from src.training import entrenar_y_evaluar

VERSION_PRUEBA = "vtest"


def _datos_sinteticos(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Clientes inventados con las 19 variables del proyecto y la variable objetivo."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    mensual = rng.uniform(18, 118, n).round(2)
    total = (tenure * mensual).round(2).astype(object)
    total[tenure == 0] = " "  # cuenta sin primera factura: faltante, como en el CSV real
    contrato = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    probabilidad = 1 / (1 + np.exp(-(1.2 - 0.05 * tenure + 0.01 * mensual - 0.8 * (contrato != "Month-to-month"))))
    servicio = ["Yes", "No", "No internet service"]
    return pd.DataFrame({
        "customerID": [f"T{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(servicio, n),
        "OnlineBackup": rng.choice(servicio, n),
        "DeviceProtection": rng.choice(servicio, n),
        "TechSupport": rng.choice(servicio, n),
        "StreamingTV": rng.choice(servicio, n),
        "StreamingMovies": rng.choice(servicio, n),
        "Contract": contrato,
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": mensual,
        "TotalCharges": total,
        "Churn": np.where(rng.random(n) < probabilidad, "Yes", "No"),
    })


@pytest.fixture(scope="module")
def entorno_modelo(tmp_path_factory):
    """Entrena UNA vez un modelo pequeño y lo guarda en una carpeta temporal."""
    base = tmp_path_factory.mktemp("api")
    csv = base / "datos.csv"
    _datos_sinteticos().to_csv(csv, index=False)
    X, y = cargar_datos_etiquetados(csv, "Churn")
    resultado = entrenar_y_evaluar(X, y, seed=42, test_size=0.2, cv_folds=0)
    models_dir = base / "models"
    guardar_version(models_dir, VERSION_PRUEBA, "Churn", 42, csv, len(X), resultado)
    return {"csv": csv, "models_dir": models_dir}


@pytest.fixture
def cliente(entorno_modelo, monkeypatch):
    """Cliente de pruebas de la API apuntando al modelo pequeño."""
    monkeypatch.setenv("MODELS_DIR", str(entorno_modelo["models_dir"]))
    monkeypatch.setenv("MODEL_VERSION", VERSION_PRUEBA)
    return TestClient(api.app)


#  pruebas mínimas (3)
def test_health_responde_200(cliente):
    """/health responde 200 e informa que la API está en ejecución."""
    r = cliente.get("/health")
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["estado"] == "ok"
    assert cuerpo["version_configurada"] == VERSION_PRUEBA
    assert cuerpo["modelo_disponible"] is True


def test_entrada_valida_devuelve_prediccion_y_version(cliente, entorno_modelo):
    """Una entrada válida devuelve la predicción y la versión, y coincide con el modelo guardado."""
    r = cliente.post("/predict", json=EJEMPLO_CLIENTE)
    assert r.status_code == 200
    cuerpo = r.json()
    assert {"prediccion", "clase", "probabilidad_churn", "version"} <= set(cuerpo)
    assert cuerpo["version"] == VERSION_PRUEBA
    assert cuerpo["prediccion"] in ("Churn", "No churn")
    assert 0.0 <= cuerpo["probabilidad_churn"] <= 1.0

    # Misma probabilidad que el artefacto guardado (con tolerancia por el redondeo a 4 decimales).
    paquete = cargar_modelo(entorno_modelo["models_dir"] / VERSION_PRUEBA)
    X = preparar_features(pd.DataFrame([EJEMPLO_CLIENTE]))
    esperada = paquete["pipeline"].predict_proba(X)[0, 1]
    assert cuerpo["probabilidad_churn"] == pytest.approx(esperada, abs=1e-4)


def test_entrada_sin_campo_obligatorio_se_rechaza(cliente):
    """Si falta una variable obligatoria: 422 y el mensaje nombra el campo."""
    entrada = {k: v for k, v in EJEMPLO_CLIENTE.items() if k != "Contract"}
    r = cliente.post("/predict", json=entrada)
    assert r.status_code == 422
    cuerpo = r.json()
    assert "Contract" in cuerpo["mensaje"]
    assert cuerpo["detalles"] == [{"campo": "Contract", "problema": "campo obligatorio ausente"}]


#  pruebas adicionales
def test_tipo_incompatible_se_rechaza_nombrando_el_campo(cliente):
    r = cliente.post("/predict", json={**EJEMPLO_CLIENTE, "tenure": "abc"})
    assert r.status_code == 422
    assert "tenure" in r.json()["mensaje"]


def test_valor_nulo_permitido_usa_la_regla_guardada(cliente):
    """Una cuenta sin primera factura (TotalCharges null) se predice con la regla de la Fase 2."""
    r = cliente.post("/predict", json={**EJEMPLO_CLIENTE, "tenure": 0, "TotalCharges": None})
    assert r.status_code == 200
    assert r.json()["prediccion"] in ("Churn", "No churn")


def test_categoria_no_vista_se_predice_con_aviso(cliente):
    r = cliente.post("/predict", json={**EJEMPLO_CLIENTE, "Contract": "Trimestral"})
    assert r.status_code == 200
    assert any("Trimestral" in aviso for aviso in r.json()["avisos"])


def test_modelo_no_disponible_devuelve_503_y_lista_versiones(cliente, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v99")
    r = cliente.post("/predict", json=EJEMPLO_CLIENTE)
    assert r.status_code == 503
    assert VERSION_PRUEBA in r.json()["mensaje"]  # sugiere una versión disponible
    assert cliente.get("/health").json()["modelo_disponible"] is False  # /health sigue en 200


def test_train_guarda_versiones_nuevas_sin_tocar_las_anteriores(entorno_modelo, tmp_path, monkeypatch):
    """/train crea v1, luego v2, y el modelo anterior queda idéntico (se usan carpetas temporales)."""
    models_dir = tmp_path / "models"
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("DATA_PATH", str(entorno_modelo["csv"]))
    monkeypatch.setenv("CV_FOLDS", "2")
    c = TestClient(api.app)

    r1 = c.post("/train")
    assert r1.status_code == 201
    assert r1.json()["version"] == "v1"
    modelo_v1 = Path(r1.json()["archivos"]["modelo"])
    assert modelo_v1.is_file()
    huella = hashlib.sha256(modelo_v1.read_bytes()).hexdigest()

    r2 = c.post("/train", json={"seed": 7})
    assert r2.status_code == 201
    assert r2.json()["version"] == "v2"
    assert hashlib.sha256(modelo_v1.read_bytes()).hexdigest() == huella
    for nombre in ("modelo.joblib", "metrics.json", "metadata.json"):
        assert (models_dir / "v2" / nombre).is_file()