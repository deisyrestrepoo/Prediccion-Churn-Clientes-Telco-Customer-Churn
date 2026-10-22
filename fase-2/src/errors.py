"""Errores previsibles de la aplicación.

Los scripts los capturan, muestran un mensaje comprensible y terminan con
código de salida distinto de cero.
"""


class ErrorAplicacion(Exception):
    """Clase base de los errores previsibles."""


class DatosInvalidosError(ErrorAplicacion):
    """Archivo de datos ausente, ilegible, con columnas faltantes o tipos incompatibles."""


class VersionExistenteError(ErrorAplicacion):
    """Ya existe una versión del modelo con ese identificador."""


class ModeloNoEncontradoError(ErrorAplicacion):
    """El artefacto del modelo no existe o no se puede cargar."""