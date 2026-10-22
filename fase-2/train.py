"""Entrena el modelo de churn y guarda una versión identificable.

Ejemplo (desde fase-2, en una sola línea):
    python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn \
        --output-dir models --version v1 --seed 42
"""
from __future__ import annotations

import argparse
import sys

from src.config import TARGET_DEFAULT
from src.data import cargar_datos_etiquetados
from src.errors import ErrorAplicacion
from src.persistence import guardar_version, validar_identificador, verificar_version_libre
from src.training import entrenar_y_evaluar


def crear_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entrena el modelo de predicción de churn y guarda una versión en --output-dir."
    )
    p.add_argument("--data", required=True, help="Ruta del CSV con datos etiquetados.")
    p.add_argument("--target", default=TARGET_DEFAULT,
                   help=f"Nombre de la variable objetivo (por defecto: {TARGET_DEFAULT}).")
    p.add_argument("--output-dir", default="models",
                   help="Directorio donde se guardan las versiones (por defecto: models).")
    p.add_argument("--version", required=True,
                   help="Identificador único de la versión (ej.: v1). No se sobrescribe.")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (por defecto: 42).")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Proporción para prueba, entre 0 y 1 (por defecto: 0.2).")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Particiones de validación cruzada sobre entrenamiento; 0 la omite (por defecto: 5).")
    return p


def _imprimir_resumen(resultado: dict, rutas: dict, version: str) -> None:
    part = resultado["particion"]
    print(f"\n=== Entrenamiento finalizado (versión {version}) ===")
    print(f"Entrenamiento: {part['filas_entrenamiento']} filas | Prueba: {part['filas_prueba']} filas")
    print(f"\n{'Métrica':<11}{'Modelo base':>13}{'Reg. Logística':>16}   (conjunto de prueba)")
    for m in ("roc_auc", "recall", "precision", "f1", "accuracy"):
        print(f"{m:<11}{resultado['metricas_base'][m]:>13.4f}{resultado['metricas_modelo'][m]:>16.4f}")
    cv = resultado["validacion_cruzada"]
    if cv:
        print(f"\nValidación cruzada ({cv['folds']} folds, solo entrenamiento): "
              f"ROC-AUC = {cv['roc_auc']['media']:.4f} ± {cv['roc_auc']['desviacion']:.4f}")
    print("\nArchivos generados:")
    for nombre, ruta in rutas.items():
        print(f"  {nombre:<9}: {ruta}")


def main(argv=None) -> int:
    args = crear_parser().parse_args(argv)
    try:
        validar_identificador(args.version)
        verificar_version_libre(args.output_dir, args.version)  # falla antes de entrenar

        X, y = cargar_datos_etiquetados(args.data, args.target)
        resultado = entrenar_y_evaluar(X, y, args.seed, args.test_size, args.cv_folds)
        rutas = guardar_version(args.output_dir, args.version, args.target,
                                args.seed, args.data, len(X), resultado)
    except ErrorAplicacion as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _imprimir_resumen(resultado, rutas, args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())