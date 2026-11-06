"""Construcción del preprocesamiento y de los modelos (sin ajustar)."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY_NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def construir_preprocesador() -> ColumnTransformer:
    """Imputación + escalamiento + codificación, todo aprendido solo con el entrenamiento.

    - Numéricas: mediana (TotalCharges tiene nulos) y StandardScaler.
    - Binaria numérica (SeniorCitizen): valor más frecuente si hubiera nulos.
    - Categóricas: categoría más frecuente si hubiera nulos y One-Hot.
      handle_unknown="ignore": una categoría no vista en el entrenamiento queda
      con todas sus columnas en 0 (no falla ni descarta la fila).
    """
    numericas = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
    ])
    binarias = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
    ])
    categoricas = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numericas", numericas, NUMERIC_COLUMNS),
            ("binarias", binarias, BINARY_NUMERIC_COLUMNS),
            ("categoricas", categoricas, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def construir_modelo(seed: int) -> Pipeline:
    """Pipeline completo: preprocesamiento + Regresión Logística (igual enfoque que la Fase 1)."""
    return Pipeline([
        ("preprocesamiento", construir_preprocesador()),
        ("clasificador", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )),
    ])


def construir_modelo_base(seed: int) -> DummyClassifier:
    """Modelo base de la Fase 1: predice siempre la clase más frecuente."""
    return DummyClassifier(strategy="most_frequent", random_state=seed)