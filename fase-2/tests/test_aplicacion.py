"""Pruebas de la aplicación de la Fase 2 (train.py, predict.py y versionamiento).

Ejecutar desde fase-2:
    python -m pytest

Todas las pruebas usan datos sintéticos pequeños (no necesitan el CSV de Kaggle),
no usan la red y escriben solo en carpetas temporales (tmp_path).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Permite importar src, train y predict aunque se ejecute `pytest` en vez de `python -m pytest`.
FASE2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FASE2))

import predict  # noqa: E402
import train  # noqa: E402
from src.config import (  # noqa: E402
    ARTIFACT_NAME,
    FEATURE_COLUMNS,
    METADATA_NAME,
    METRICS_NAME,
)
from src.data import preparar_features  # noqa: E402
from src.errors import DatosInvalidosError  # noqa: E402
from src.persistence import cargar_modelo  # noqa: E402

# ---------------------------------------------------------------------------
# Datos sintéticos y utilidades
# ---------------------------------------------------------------------------


def _crear_datos(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Clientes inventados con el mismo esquema del dataset Telco (incluye 'Churn')."""
    rng = np.random.default_rng(seed)
    servicio = ["Yes", "No", "No internet service"]
    df = pd.DataFrame({
        "customerID": [f"T-{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": rng.integers(0, 72, n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(servicio, n),
        "OnlineBackup": rng.choice(servicio, n),
        "DeviceProtection": rng.choice(servicio, n),
        "TechSupport": rng.choice(servicio, n),
        "StreamingTV": rng.choice(servicio, n),
        "StreamingMovies": rng.choice(servicio, n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": rng.uniform(18, 118, n).round(2),
    })
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "  # como en el dataset real
    prob = 1 / (1 + np.exp(-(1.0 - 0.04 * df["tenure"]
                             + (df["Contract"] == "Month-to-month") * 1.0)))
    df["Churn"] = np.where(rng.random(n) < prob, "Yes", "No")
    return df


def _entrenar(csv: Path, salida: Path, version: str, seed: int = 42) -> int:
    """Ejecuta train.py (la función main) y devuelve su código de salida."""
    return train.main([
        "--data", str(csv), "--target", "Churn", "--output-dir", str(salida),
        "--version", version, "--seed", str(seed),
        "--test-size", "0.25", "--cv-folds", "2",
    ])


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _huellas(carpeta: Path) -> dict:
    """Huella de cada archivo de una versión, para comprobar que no cambia."""
    return {p.name: _sha256(p) for p in sorted(carpeta.iterdir())}


@pytest.fixture()
def datos_csv(tmp_path) -> Path:
    ruta = tmp_path / "datos.csv"
    _crear_datos().to_csv(ruta, index=False)
    return ruta


# ---------------------------------------------------------------------------
# 1. Entrenamiento y almacenamiento
# ---------------------------------------------------------------------------


def test_entrenamiento_guarda_modelo_metricas_y_metadatos(datos_csv, tmp_path, capsys):
    modelos = tmp_path / "models"

    assert _entrenar(datos_csv, modelos, "v1") == 0
    carpeta = modelos / "v1"
    for nombre in (ARTIFACT_NAME, METRICS_NAME, METADATA_NAME):
        assert (carpeta / nombre).is_file(), f"falta {nombre}"

    # El artefacto se puede cargar y trae el Pipeline completo con su esquema.
    paquete = cargar_modelo(carpeta)
    assert paquete["version"] == "v1"
    assert paquete["esquema"]["variables_entrada"] == FEATURE_COLUMNS
    assert hasattr(paquete["pipeline"], "predict_proba")

    # Métricas del modelo y del modelo base, sobre el mismo conjunto de prueba.
    metricas = json.loads((carpeta / METRICS_NAME).read_text(encoding="utf-8"))
    for bloque in ("modelo", "modelo_base"):
        for metrica in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= metricas[bloque][metrica] <= 1.0
    assert metricas["modelo_base"]["roc_auc"] == pytest.approx(0.5)
    assert metricas["modelo_base"]["recall"] == 0.0

    # Metadatos mínimos exigidos.
    meta = json.loads((carpeta / METADATA_NAME).read_text(encoding="utf-8"))
    for clave in ("version", "fecha_creacion", "tipo_problema", "variable_objetivo",
                  "algoritmo", "parametros_modelo", "semilla", "variables_entrada",
                  "tipos_esperados", "datos", "particion", "metrica_principal",
                  "entorno", "codigo"):
        assert clave in meta, f"metadata.json no tiene '{clave}'"
    assert meta["version"] == "v1" and meta["semilla"] == 42
    assert meta["datos"]["observaciones"] == 300
    assert meta["particion"]["filas_entrenamiento"] + meta["particion"]["filas_prueba"] == 300
    assert meta["entorno"]["python"] == platform.python_version()

    # El resumen por consola informa las rutas generadas.
    assert "Archivos generados" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 2. Carga y predicción
# ---------------------------------------------------------------------------


def test_prediccion_una_por_fila_y_coincide_con_el_modelo_guardado(datos_csv, tmp_path):
    modelos = tmp_path / "models"
    assert _entrenar(datos_csv, modelos, "v1") == 0

    entrada = pd.read_csv(datos_csv).drop(columns=["Churn"]).head(25)
    entrada_csv = tmp_path / "entrada.csv"
    entrada.to_csv(entrada_csv, index=False)
    salida = tmp_path / "salida" / "predicciones.csv"

    codigo = predict.main([
        "--model", str(modelos / "v1"), "--input", str(entrada_csv),
        "--output", str(salida), "--probabilidades",
    ])
    assert codigo == 0

    resultado = pd.read_csv(salida)
    # Una predicción por fila, en el mismo orden y con el identificador conservado.
    assert len(resultado) == len(entrada)
    assert resultado["customerID"].tolist() == entrada["customerID"].tolist()
    assert set(resultado["prediccion"]) <= {0, 1}
    assert set(resultado["prediccion_etiqueta"]) <= {"No churn", "Churn"}
    assert resultado["probabilidad_churn"].between(0, 1).all()

    # Coincide con lo que predice el Pipeline cargado directamente desde el archivo.
    pipeline = cargar_modelo(modelos / "v1")["pipeline"]
    esperado = pipeline.predict(preparar_features(entrada))
    assert resultado["prediccion"].tolist() == esperado.tolist()

    # Y es la misma predicción del modelo recién entrenado, antes de guardarlo.
    from src.data import cargar_datos_etiquetados
    from src.training import entrenar_y_evaluar

    X, y = cargar_datos_etiquetados(datos_csv, "Churn")
    en_memoria = entrenar_y_evaluar(X, y, seed=42, test_size=0.25, cv_folds=0)["modelo"]
    assert resultado["prediccion"].tolist() == en_memoria.predict(preparar_features(entrada)).tolist()


# ---------------------------------------------------------------------------
# 3. Valores faltantes
# ---------------------------------------------------------------------------


def test_valores_faltantes_se_imputan_y_no_se_pierden_filas(datos_csv, tmp_path):
    modelos = tmp_path / "models"
    assert _entrenar(datos_csv, modelos, "v1") == 0

    entrada = pd.read_csv(datos_csv).drop(columns=["Churn"]).head(10).copy()
    entrada["TotalCharges"] = entrada["TotalCharges"].astype(object)
    entrada.loc[0, "TotalCharges"] = " "     # vacío, como en el dataset real (tenure = 0)
    entrada.loc[1, "TotalCharges"] = np.nan  # nulo
    entrada.loc[2, "MonthlyCharges"] = np.nan
    entrada.loc[3, "Contract"] = np.nan      # categórica sin valor
    entrada_csv = tmp_path / "con_nulos.csv"
    entrada.to_csv(entrada_csv, index=False)
    salida = tmp_path / "predicciones.csv"

    # Un valor nulo no es una columna ausente: no hay error y no se descarta ninguna fila.
    assert predict.main(["--model", str(modelos / "v1"), "--input", str(entrada_csv),
                         "--output", str(salida)]) == 0
    resultado = pd.read_csv(salida)
    assert len(resultado) == 10
    assert resultado["customerID"].tolist() == entrada["customerID"].tolist()
    assert resultado["prediccion"].notna().all()

    # El preprocesamiento guardado deja las variables sin faltantes (estrategia documentada).
    pipeline = cargar_modelo(modelos / "v1")["pipeline"]
    transformado = pipeline.named_steps["preprocesamiento"].transform(preparar_features(entrada))
    assert not np.isnan(transformado).any()


# ---------------------------------------------------------------------------
# 4. Entrada inválida
# ---------------------------------------------------------------------------


def test_entrada_invalida_da_error_claro_y_codigo_distinto_de_cero(datos_csv, tmp_path, capsys):
    modelos = tmp_path / "models"
    assert _entrenar(datos_csv, modelos, "v1") == 0
    capsys.readouterr()

    #a)- -nivel de función: columna ausente => error claro que nombra la columna.
    sin_columna = pd.read_csv(datos_csv).drop(columns=["Churn", "tenure"]).head(5)
    with pytest.raises(DatosInvalidosError, match="Faltan columnas requeridas: tenure"):
        preparar_features(sin_columna)

    #b-Texto en una columna numérica => error claro.
    texto = pd.read_csv(datos_csv).drop(columns=["Churn"]).head(5).copy()
    texto["MonthlyCharges"] = texto["MonthlyCharges"].astype(object)
    texto.loc[2, "MonthlyCharges"] = "abc"
    with pytest.raises(DatosInvalidosError, match="MonthlyCharges"):
        preparar_features(texto)

    #c-A nivel de línea de comandos (proceso real): código distinto de cero,
    # mensaje comprensible y SIN archivo de salida.
    entrada_mala = tmp_path / "sin_tenure.csv"
    sin_columna.to_csv(entrada_mala, index=False)
    salida = tmp_path / "salida" / "malo.csv"
    proceso = subprocess.run(
        [sys.executable, str(FASE2 / "predict.py"), "--model", str(modelos / "v1"),
         "--input", str(entrada_mala), "--output", str(salida)],
        capture_output=True, text=True, cwd=FASE2,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert proceso.returncode != 0
    assert "Error:" in proceso.stderr and "tenure" in proceso.stderr
    assert not salida.exists()

    #d- Otros errores previsibles de predict.py y train.py (por función main).
    assert predict.main(["--version", "v99", "--models-dir", str(modelos),
                         "--input", str(entrada_mala), "--output", str(salida)]) == 1
    assert predict.main(["--input", str(entrada_mala), "--output", str(salida)]) == 1
    assert predict.main(["--model", str(modelos / "v1"),
                         "--input", str(tmp_path / "no_existe.csv"), "--output", str(salida)]) == 1
    assert not salida.exists()

    sin_objetivo = tmp_path / "sin_objetivo.csv"
    pd.read_csv(datos_csv).drop(columns=["Churn"]).to_csv(sin_objetivo, index=False)
    assert _entrenar(sin_objetivo, modelos, "v2") == 1
    assert not (modelos / "v2").exists()  # una versión fallida no deja carpetas a medias
    assert "Error:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 5. Versionamiento
# ---------------------------------------------------------------------------


def test_versionamiento_conserva_la_primera_y_rechaza_ids_repetidos(datos_csv, tmp_path, capsys):
    modelos = tmp_path / "models"

    assert _entrenar(datos_csv, modelos, "v1", seed=42) == 0
    huellas_v1 = _huellas(modelos / "v1")

    # Una segunda versión conserva la primera intacta y ambas se pueden cargar.
    assert _entrenar(datos_csv, modelos, "v2", seed=7) == 0
    assert _huellas(modelos / "v1") == huellas_v1
    assert cargar_modelo(modelos / "v1")["version"] == "v1"
    assert cargar_modelo(modelos / "v2")["version"] == "v2"
    meta_v1 = json.loads((modelos / "v1" / METADATA_NAME).read_text(encoding="utf-8"))
    meta_v2 = json.loads((modelos / "v2" / METADATA_NAME).read_text(encoding="utf-8"))
    assert (meta_v1["semilla"], meta_v2["semilla"]) == (42, 7)

    # Repetir un identificador NO sobrescribe: error claro, código 1 y archivos idénticos.
    capsys.readouterr()
    assert _entrenar(datos_csv, modelos, "v1", seed=99) == 1
    assert "ya existe" in capsys.readouterr().err
    assert _huellas(modelos / "v1") == huellas_v1

    # Un identificador con caracteres no válidos (rutas) también se rechaza.
    assert _entrenar(datos_csv, modelos, "../fuera", seed=42) == 1
    assert not (tmp_path / "fuera").exists()
