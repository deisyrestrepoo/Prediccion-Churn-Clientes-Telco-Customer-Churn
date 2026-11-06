"""Genera predicciones de churn con una versión guardada del modelo (no reentrena).

Ejemplos (desde fase-2, en una sola línea):
    python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
    python predict.py --version v2 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv --probabilidades
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, ID_COLUMN, LABEL_NAMES
from src.data import leer_csv, preparar_features
from src.errors import DatosInvalidosError, ErrorAplicacion
from src.persistence import cargar_modelo, ruta_version


def crear_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predice el churn de clientes con un modelo guardado. No reentrena ni reajusta nada."
    )
    origen = p.add_argument_group("modelo (indique --model o --version)")
    origen.add_argument("--model", help="Ruta de modelo.joblib o de la carpeta de la versión (ej.: models/v1).")
    origen.add_argument("--version", help="Identificador de la versión dentro de --models-dir (ej.: v1).")
    origen.add_argument("--models-dir", default="models",
                        help="Carpeta con las versiones, usada con --version (por defecto: models).")
    p.add_argument("--input", required=True, help="CSV con los clientes a predecir (sin la columna objetivo).")
    p.add_argument("--output", required=True, help="Ruta del CSV de salida con las predicciones.")
    p.add_argument("--probabilidades", action="store_true",
                   help="Agrega la columna probabilidad_churn (probabilidad de la clase 1).")
    return p


def _resolver_modelo(args) -> Path:
    if bool(args.model) == bool(args.version):
        raise DatosInvalidosError("Indique exactamente una opción: --model o --version.")
    if args.model:
        return Path(args.model)
    return ruta_version(args.models_dir, args.version)


def _verificar_esquema(paquete: dict) -> None:
    """El esquema guardado con el modelo debe coincidir con el que espera este código."""
    guardado = list(paquete["esquema"]["variables_entrada"])
    if guardado != FEATURE_COLUMNS:
        raise DatosInvalidosError(
            "El esquema del modelo guardado no coincide con el esquema actual del código "
            "(el modelo se entrenó con otras variables)."
        )


def _avisar_categorias_nuevas(pipeline, X: pd.DataFrame) -> None:
    """Informa (sin fallar) las categorías que el modelo no vio al entrenar.

    El OneHotEncoder usa handle_unknown="ignore": esas filas se predicen igual,
    con todas las columnas de esa variable en 0.
    """
    try:
        codificador = pipeline.named_steps["preprocesamiento"] \
            .named_transformers_["categoricas"].named_steps["onehot"]
        # El imputador entrega un arreglo sin nombres: las categorías van en el orden
        # de CATEGORICAL_COLUMNS, que es el orden en que se entrenó el ColumnTransformer.
        conocidas = dict(zip(CATEGORICAL_COLUMNS, codificador.categories_))
    except (KeyError, AttributeError):
        return
    for col, cats in conocidas.items():
        nuevas = sorted(set(X[col].dropna().unique()) - set(cats))
        if nuevas:
            print(f"Aviso: '{col}' trae categorías no vistas en el entrenamiento {nuevas}; "
                  "se predicen igual (sin esa categoría).", file=sys.stderr)


def predecir(paquete: dict, df: pd.DataFrame, con_probabilidad: bool) -> pd.DataFrame:
    """Una fila de salida por cada fila de entrada, en el mismo orden."""
    X = preparar_features(df)          # valida columnas y tipos; ordena como en el entrenamiento
    pipeline = paquete["pipeline"]
    _avisar_categorias_nuevas(pipeline, X)

    clases = pipeline.predict(X)
    etiquetas = paquete.get("etiquetas", LABEL_NAMES)

    salida = pd.DataFrame(index=df.index)
    if ID_COLUMN in df.columns:
        salida[ID_COLUMN] = df[ID_COLUMN]
    salida["prediccion"] = np.asarray(clases).astype(int)
    salida["prediccion_etiqueta"] = [etiquetas[int(c)] for c in clases]
    if con_probabilidad:
        salida["probabilidad_churn"] = pipeline.predict_proba(X)[:, 1].round(4)
    return salida


def main(argv=None) -> int:
    args = crear_parser().parse_args(argv)
    try:
        paquete = cargar_modelo(_resolver_modelo(args))
        _verificar_esquema(paquete)
        df = leer_csv(args.input)
        resultado = predecir(paquete, df, args.probabilidades)

        # Solo se escribe el archivo cuando TODO salió bien: nunca queda uno a medias.
        destino = Path(args.output)
        try:
            destino.parent.mkdir(parents=True, exist_ok=True)
            resultado.to_csv(destino, index=False)
        except OSError as exc:
            raise DatosInvalidosError(f"No se pudo escribir la salida {destino}: {exc}") from exc
    except ErrorAplicacion as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    n_churn = int((resultado["prediccion"] == 1).sum())
    print(f"Modelo versión {paquete['version']}: {len(resultado)} predicciones "
          f"({n_churn} con churn, {len(resultado) - n_churn} sin churn).")
    print(f"Archivo generado: {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
