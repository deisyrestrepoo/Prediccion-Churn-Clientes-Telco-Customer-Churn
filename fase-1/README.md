# Predicción de Churn de Clientes — Telco Customer Churn

Proyecto Integrador — Fase 1: Modelo Predictivo
Modelos y Simulación de Sistemas I — 2026 II

## Integrantes del equipo
- EDGAR ANDRES GARZON MARIN
- DEISY TATIANA RESTREPO OSPINA

## Descripción del problema
Se busca predecir si un cliente de una empresa de telecomunicaciones abandonará el
servicio (churn) a partir de sus características demográficas, los servicios
contratados y su información de facturación. Es un problema de **clasificación
binaria** de gran relevancia para estrategias de retención.

## Fuente del conjunto de datos
Dataset "Telco Customer Churn" de Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
(7.043 observaciones, 21 variables; 26,5 % de clientes con churn; valores faltantes
únicamente en `TotalCharges`: 11 filas, 0,16 %).

## Objetivo del modelo
Predecir la variable `Churn` (Yes/No) para identificar clientes con alto riesgo de
abandono antes de que ocurra.

## Algoritmo utilizado
**Regresión Logística** (`scikit-learn`) dentro de un `Pipeline` con `StandardScaler`
aplicado a `tenure`, `MonthlyCharges` y `TotalCharges`, y `class_weight="balanced"`
para compensar el desbalance de clases. Modelo base de comparación:
`DummyClassifier(strategy="most_frequent")`.

## Métrica empleada
**ROC-AUC** como métrica principal (no depende del umbral ni se distorsiona por el
desbalance de clases), complementada con Recall, Precision, F1-score y Accuracy.
Validación cruzada estratificada de 5 folds sobre el conjunto de entrenamiento y
evaluación final única sobre el conjunto de prueba (split 80/20 estratificado,
`random_state=42`).

## Principales resultados obtenidos
Resultados en el conjunto de prueba:

| Métrica   | Modelo base (Dummy) | Regresión Logística |
|-----------|---------------------|---------------------|
| Accuracy  | 0.7342              | 0.7257              |
| Precision | 0.0000              | 0.4901              |
| Recall    | 0.0000              | 0.7941              |
| F1-score  | 0.0000              | 0.6061              |
| ROC-AUC   | 0.5000              | 0.8353              |

El modelo supera ampliamente al modelo base en Recall, F1 y ROC-AUC. Las variables más
influyentes fueron [tenure, tipo de contrato, servicio de fibra óptica]. Se evitó la
fuga de información ajustando el escalador solo con los datos de entrenamiento
(dentro del Pipeline) y usando el conjunto de prueba una única vez.

## Estructura de la carpeta
- `notebook.ipynb`: análisis completo (EDA, preparación, modelo base, modelo predictivo, evaluación, guardado).
- `modelo.joblib`: Pipeline entrenado (escalador + Regresión Logística).
- `requirements.txt`: versiones de las librerías usadas.
- `README.md`: este archivo.

## Instrucciones para ejecutar el notebook
1. Abrir `notebook.ipynb` en Google Colab (Archivo → Abrir cuaderno → GitHub).
2. Crear un token en Kaggle (Settings → API → Create New Token) y guardarlo en
   Colab Secrets (ícono de llave, panel izquierdo) con el nombre `KAGGLE_API_TOKEN`,
   activando el acceso al notebook.
3. Ejecutar todo: *Entorno de ejecución → Ejecutar todas*. No requiere intervención manual.
4. Al final el notebook guarda el modelo como `modelo.joblib` (en Colab queda en
   `/content/`; descargarlo desde el panel de archivos si se desea conservarlo).

Versiones usadas: ver `requirements.txt`.
