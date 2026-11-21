"""Esquema del problema (Telco Customer Churn) y constantes compartidas."""

DATA_SOURCE = "Kaggle: blastchar/telco-customer-churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)"

TARGET_DEFAULT = "Churn"
ID_COLUMN = "customerID"  # no es predictora; si viene en la entrada, se conserva en la salida

# Etiquetas de la variable objetivo en el CSV -> clase numérica
TARGET_MAPPING = {"No": 0, "Yes": 1}
LABEL_NAMES = {0: "No churn", 1: "Churn"}

# Variables predictoras agrupadas por tratamiento
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_NUMERIC_COLUMNS = ["SeniorCitizen"]  # ya viene como 0/1
CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + BINARY_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# Archivos de cada versión
ARTIFACT_NAME = "modelo.joblib"
METRICS_NAME = "metrics.json"
METADATA_NAME = "metadata.json"