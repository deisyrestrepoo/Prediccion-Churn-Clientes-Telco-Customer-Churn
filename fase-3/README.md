# Fase 3 — API REST de predicción de churn

**Curso:** Modelos y Simulación de Sistemas I · Proyecto Integrador 2026-2 · Universidad de Antioquia
**Docente:** Andrés Parra
**Integrantes:** Deisy Tatiana Restrepo Ospina, Edgar Andrés Garzón Marín

## 1. Relación con la Fase 2

La Fase 3 expone por HTTP el modelo de la Fase 2 (clasificación de churn de clientes Telco): consultar una predicción, comprobar que el servicio está activo y entrenar una nueva versión. Se reutilizan la lógica de `fase-2/src/`, el esquema de entradas y el preprocesamiento guardado con cada versión. `fase-1/` y `fase-2/` se conservan sin cambios.

### Estado de la fase (por Pull Requests)

| PR / rama | Contenido | Estado |
|---|---|---|
| `feature/api-base` | Esqueleto de `fase-3/`, configuración y `GET /health` | Integrado en `develop` |
| `feature/api-predict` | `POST /predict` con validación de entradas y errores 422 y 503 | **Este PR** |
| `feature/api-train` | `POST /train` con versión nueva automática | Pendiente |
| `feature/api-docker-tests` | Dockerfile, pruebas con pytest, cliente de prueba y documentación completa | Pendiente |

## 2. Estructura

```
fase-3/
├── api.py             # Aplicación FastAPI (rutas y manejo de errores)
├── requirements.txt   # Dependencias con versiones explícitas
├── src/
│   ├── config.py, data.py, errors.py, persistence.py,
│   │   preprocessing.py, training.py   # Copia sin cambios de fase-2/src/
│   ├── settings.py    # Configuración por variables de entorno
│   ├── schemas.py     # Entrada (19 variables) y salida de /predict
│   └── servicio.py    # Carga de la versión configurada y predicción de un registro
├── models/            # Versiones guardadas: v1/ y v2/ (copiadas de fase-2/models/)
├── ejemplos/          # JSON de ejemplo para probar /predict
└── README.md
```

**Origen del código.** Los archivos de `src/` indicados como «copia» y las versiones `v1` y `v2` son idénticos a los de `fase-2`, para que `fase-3` se pueda construir y ejecutar por sí sola (incluido Docker). `settings.py`, `schemas.py` y `servicio.py` son nuevos de esta fase.

## 3. Configuración

La API se configura con variables de entorno, todas opcionales:

| Variable | Por defecto | Para qué sirve |
|---|---|---|
| `MODEL_VERSION` | `v1` | Versión del modelo que usa `/predict`. Para volver a la anterior basta cambiar este valor y reiniciar |
| `MODELS_DIR` | `models` | Carpeta con las versiones guardadas |
| `DATA_PATH` | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | CSV con datos etiquetados que usará `/train` |
| `TARGET` | `Churn` | Variable objetivo de `/train` |
| `SEED`, `TEST_SIZE`, `CV_FOLDS` | `42`, `0.2`, `5` | Parámetros de `/train` (solo se leen al entrenar) |

### Cómo seleccionar una versión disponible

Las versiones disponibles son las carpetas de `MODELS_DIR` que contienen `modelo.joblib` (hoy `v1` y `v2`). Se elige con `MODEL_VERSION` y se reinicia la API:

```powershell
$env:MODEL_VERSION = "v2"; uvicorn api:app --host 127.0.0.1 --port 8000
```

(En Bash: `MODEL_VERSION=v2 uvicorn api:app --host 127.0.0.1 --port 8000`.) Si la versión configurada no existe, `/predict` responde 503 y el mensaje lista las versiones disponibles.

## 4. Instalación y ejecución local

Python 3.12 (el mismo de la Fase 2). Desde la raíz del repositorio, con el entorno virtual activo (PowerShell):

```powershell
cd fase-3
pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8000
```

La documentación interactiva queda en http://127.0.0.1:8000/docs (incluye un ejemplo de solicitud para `/predict` y permite probarla desde el navegador).

## 5. Rutas

### `GET /health`

Indica que la API está en ejecución y qué versión tiene configurada. Responde 200 aunque el modelo no esté disponible: `/health` describe el servicio, no la predicción.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

```json
{"estado": "ok", "mensaje": "API en ejecución", "version_configurada": "v1", "modelo_disponible": true}
```

### `POST /predict`

Recibe las **19 variables predictoras** de un cliente en JSON y devuelve la predicción con la versión del modelo utilizada. No necesita la variable objetivo (`Churn`).

**Entrada.** Las claves conservan los nombres y tipos del entrenamiento: un número se envía como número (sin comillas) y una categoría como texto.

| Tipo | Variables |
|---|---|
| Texto (categóricas) | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Entero 0 o 1 | `SeniorCitizen` |
| Entero ≥ 0 | `tenure` |
| Número ≥ 0 | `MonthlyCharges`, `TotalCharges` |

- Las 19 claves son **obligatorias**: si falta una, la respuesta es 422 y nombra el campo.
- Cualquier valor puede ser `null` (dato faltante): se aplica la regla guardada en la Fase 2 (mediana del entrenamiento en las numéricas y categoría más frecuente en las demás). Un `TotalCharges` `null` corresponde a una cuenta sin primera factura.
- Las claves adicionales (`customerID`, `Churn`, ...) se ignoran.
- Una categoría que el modelo no vio al entrenar no es un error: se predice igual y se informa en `avisos`.

**Ejemplo** (con el archivo `ejemplos/cliente_valido.json`):

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@ejemplos/cliente_valido.json"
```

Equivalente en PowerShell puro:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body (Get-Content ejemplos/cliente_valido.json -Raw)
```

Respuesta (HTTP 200):

```json
{"prediccion": "Churn", "clase": 1, "probabilidad_churn": 0.8057, "version": "v1", "avisos": []}
```

| Campo | Significado |
|---|---|
| `prediccion` | Etiqueta: `Churn` (el cliente probablemente abandonará) o `No churn` |
| `clase` | `1` o `0`, equivalente a la etiqueta |
| `probabilidad_churn` | Probabilidad de la clase 1; el umbral de decisión es 0.5 |
| `version` | Versión del modelo que produjo la predicción |
| `avisos` | Avisos no bloqueantes (categorías no vistas) |

Con `ejemplos/cliente_sin_primera_factura.json` (`tenure` 0 y `TotalCharges` `null`) la respuesta es 200 con la regla de imputación guardada.

### Errores

Todos los errores usan el mismo formato: `{"error": ..., "mensaje": ..., "detalles": [...]}`.

| Situación | Código | Ejemplo de mensaje |
|---|---|---|
| Falta una variable obligatoria | 422 | `Contract: campo obligatorio ausente` |
| Tipo incompatible | 422 | `tenure: debe ser un número entero (sin comillas)` |
| Valor fuera de rango | 422 | `SeniorCitizen: valor no permitido (se esperaba 0 o 1)` |
| Cuerpo que no es JSON | 422 | `cuerpo: no es un JSON válido` |
| Versión configurada no disponible | 503 | `El modelo de la versión 'v99' no está disponible. Versiones disponibles en 'models': v1, v2. Configure la variable de entorno MODEL_VERSION con una de ellas y reinicie la API.` |
| Variable de entorno inválida | 500 | `La variable de entorno SEED='abc' no es válida (se esperaba int).` |

Prueba del error de validación (`ejemplos/cliente_sin_contract.json`):

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@ejemplos/cliente_sin_contract.json"
```

```json
{"error": "Entrada inválida", "mensaje": "Contract: campo obligatorio ausente", "detalles": [{"campo": "Contract", "problema": "campo obligatorio ausente"}]}
```

## 6. Pendiente en los siguientes PR

- `POST /train` con identificador de versión disponible y sin tocar las versiones anteriores.
- Dockerfile actualizado, pruebas con pytest y cliente de prueba.
- Comandos de Docker con montajes de datos y modelos.