"""Separación de datos, entrenamiento y evaluación."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .errors import DatosInvalidosError
from .preprocessing import construir_modelo, construir_modelo_base

METRICA_PRINCIPAL = "roc_auc"
METRICAS = ["roc_auc", "recall", "precision", "f1", "accuracy"]


def separar_datos(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    """Partición estratificada ANTES de ajustar cualquier transformación.

    No hay grupos naturales (cada fila es un cliente distinto), por eso basta
    estratificar por la variable objetivo.
    """
    if not 0 < test_size < 1:
        raise DatosInvalidosError("--test-size debe estar entre 0 y 1 (sin incluirlos).")
    if y.value_counts().min() < 2:
        raise DatosInvalidosError(
            "Cada clase necesita al menos 2 observaciones para separar entrenamiento y prueba."
        )
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError as exc:
        raise DatosInvalidosError(f"No se pudo separar entrenamiento y prueba: {exc}") from exc


def calcular_metricas(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _evaluar(modelo, X_test, y_test) -> dict:
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    resultado = calcular_metricas(y_test, y_pred, y_proba)
    resultado["matriz_confusion"] = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    return resultado


def validacion_cruzada(modelo, X_train, y_train, folds: int, seed: int) -> dict:
    """CV estratificada solo sobre entrenamiento; el pipeline se reajusta en cada partición."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    try:
        res = cross_validate(clone(modelo), X_train, y_train, cv=cv, scoring=METRICAS, n_jobs=1)
    except ValueError as exc:
        raise DatosInvalidosError(f"No se pudo ejecutar la validación cruzada: {exc}") from exc
    salida = {"folds": folds}
    for m in METRICAS:
        valores = res[f"test_{m}"]
        salida[m] = {"media": float(np.mean(valores)), "desviacion": float(np.std(valores))}
    return salida


def entrenar_y_evaluar(X, y, seed: int, test_size: float, cv_folds: int) -> dict:
    """Flujo completo: separar -> (CV) -> entrenar -> evaluar modelo y modelo base.

    Ambos modelos usan la misma partición y las mismas métricas (comparación de la Fase 1).
    """
    X_train, X_test, y_train, y_test = separar_datos(X, y, test_size, seed)

    modelo = construir_modelo(seed)
    cv = validacion_cruzada(modelo, X_train, y_train, cv_folds, seed) if cv_folds > 0 else None
    modelo.fit(X_train, y_train)

    base = construir_modelo_base(seed)
    base.fit(X_train, y_train)

    return {
        "modelo": modelo,
        "metricas_modelo": _evaluar(modelo, X_test, y_test),
        "metricas_base": _evaluar(base, X_test, y_test),
        "validacion_cruzada": cv,
        "particion": {
            "estrategia": "train_test_split estratificado por la variable objetivo",
            "test_size": test_size,
            "seed": seed,
            "filas_entrenamiento": int(len(X_train)),
            "filas_prueba": int(len(X_test)),
            "proporcion_churn_entrenamiento": float(y_train.mean()),
            "proporcion_churn_prueba": float(y_test.mean()),
        },
    }
