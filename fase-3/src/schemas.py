"""Esquemas de entrada y salida de la API.

La entrada reproduce las 19 variables predictoras del proyecto con los mismos nombres y
tipos que se usaron al entrenar (ver src/config.py): un número se envía como número
(sin comillas) y una categoría como texto.

Reglas:
- Las 19 claves son OBLIGATORIAS: si falta una, la API responde 422 y nombra el campo.
- Cada valor puede ser null: es un dato faltante y se trata con la regla guardada en la
  Fase 2 (mediana del entrenamiento en las numéricas, categoría más frecuente en las demás).
- Un tipo incompatible (por ejemplo "tenure": "abc") responde 422.
- Las claves adicionales (customerID, Churn, etc.) se ignoran: no hace falta enviar la
  variable objetivo.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

Numero = Union[StrictFloat, StrictInt]

EJEMPLO_CLIENTE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


class ClienteEntrada(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={"example": EJEMPLO_CLIENTE})

    # Categóricas (texto)
    gender: Optional[str]
    Partner: Optional[str]
    Dependents: Optional[str]
    PhoneService: Optional[str]
    MultipleLines: Optional[str]
    InternetService: Optional[str]
    OnlineSecurity: Optional[str]
    OnlineBackup: Optional[str]
    DeviceProtection: Optional[str]
    TechSupport: Optional[str]
    StreamingTV: Optional[str]
    StreamingMovies: Optional[str]
    Contract: Optional[str]
    PaperlessBilling: Optional[str]
    PaymentMethod: Optional[str]
    # Binaria numérica (0 o 1)
    SeniorCitizen: Optional[Literal[0, 1]]
    # Numéricas
    tenure: Annotated[Optional[StrictInt], Field(ge=0)]
    MonthlyCharges: Annotated[Optional[Numero], Field(ge=0)]
    TotalCharges: Annotated[Optional[Numero], Field(ge=0)]


class PrediccionSalida(BaseModel):
    prediccion: str = Field(description="Etiqueta predicha: 'Churn' o 'No churn'.")
    clase: int = Field(description="1 = el cliente probablemente abandonará; 0 = se quedará.")
    probabilidad_churn: float = Field(description="Probabilidad de la clase 1 (umbral de decisión 0.5).")
    version: str = Field(description="Versión del modelo que produjo la predicción.")
    avisos: list[str] = Field(default_factory=list,
                              description="Avisos no bloqueantes, p. ej. categorías no vistas en el entrenamiento.")