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
| `feature/api-predict` | `POST /predict` con validación de entradas y errores 422 y 503 | Integrado en `develop` |
| `feature/api-train` | `POST /train` con versión nueva automática | **Este PR** |
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
| `DATA_PATH` | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | CSV con datos etiquetados que usa `/train` (ver «Datos para `/train`») |
| `TARGET` | `Churn` | Variable objetivo de `/train` |
| `SEED`, `TEST_SIZE`, `CV_FOLDS` | `42`, `0.2`, `5` | Parámetros de `/train` (los de la Fase 2; solo se leen al entrenar) |

### Cómo seleccionar una versión disponible

Las versiones disponibles son las carpetas de `MODELS_DIR` que contienen `modelo.joblib` (hoy `v1` y `v2`). Se elige con `MODEL_VERSION` y se reinicia la API:

```powershell
$env:MODEL_VERSION = "v2"; uvicorn api:app --host 127.0.0.1 --port 8000
```

(En Bash: `MODEL_VERSION=v2 uvicorn api:app --host 127.0.0.1 --port 8000`.) Si la versión configurada no existe, `/predict` responde 503 y el mensaje lista las versiones disponibles.

### Datos para `/train`

El CSV completo **no está en el repositorio**. Para poder entrenar:

1. Descargar *Telco Customer Churn* desde Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Dejarlo en `fase-3/data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (la carpeta `data/` está en el `.gitignore`), o apuntar `DATA_PATH` a otra ubicación.

Sin el CSV, `/train` responde 503 con ese mismo aviso. `/health` y `/predict` no lo necesitan.

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

### `POST /train`

Entrena con los datos y la configuración de la Fase 2 (la misma lógica de `train.py`: partición 80/20 estratificada **antes** de ajustar transformaciones, validación cruzada de 5 folds y modelo base) y guarda una **versión nueva** en `MODELS_DIR` con su modelo, métricas y metadatos. Las versiones anteriores no se tocan.

- **Cuerpo:** puede ir vacío. Opcionalmente `{"seed": 7}` cambia la semilla de la partición (por defecto `SEED`, que es 42).
- **Identificador:** `v` + (el mayor número existente + 1). Con `v1` y `v2`, la nueva es `v3`. Nunca sobrescribe una versión existente (`v-docker` y otros nombres no numéricos se ignoran para la numeración).
- **Un entrenamiento a la vez:** una segunda solicitud mientras hay otro en curso responde 409.
- **Duración:** de unos segundos a un par de minutos según el equipo (incluye la validación cruzada); la respuesta llega cuando los artefactos ya están guardados.
- **La versión nueva no se activa sola:** `/predict` sigue usando `MODEL_VERSION` hasta que se cambie esa variable y se reinicie la API.

```powershell
curl.exe -X POST http://127.0.0.1:8000/train
# PowerShell puro: Invoke-RestMethod -Method Post http://127.0.0.1:8000/train
```

Respuesta (HTTP 201):

```json
{
  "version": "v3",
  "metrica_principal": "roc_auc",
  "seed": 42,
  "observaciones": 7043,
  "filas_entrenamiento": 5634,
  "filas_prueba": 1409,
  "metricas": {"roc_auc": 0.8416, "recall": 0.7834, "precision": 0.5043, "f1": 0.6136, "accuracy": 0.7381},
  "metricas_modelo_base": {"roc_auc": 0.5, "recall": 0.0, "precision": 0.0, "f1": 0.0, "accuracy": 0.7346},
  "archivos": {"modelo": "models/v3/modelo.joblib", "metricas": "models/v3/metrics.json", "metadata": "models/v3/metadata.json"},
  "version_activa": "v1",
  "mensaje": "Versión v3 guardada; las anteriores se conservan. /predict sigue usando 'v1'. Para activar la nueva, configure MODEL_VERSION=v3 y reinicie la API."
}
```

Con la semilla 42 los resultados coinciden con los de `v1` (ROC-AUC 0.8416); con `{"seed": 7}` coinciden con los de `v2` (0.8567).

**Comprobar la versión nueva en el almacenamiento:** `ls models` (Bash) o `dir models` (PowerShell) debe mostrar `v1`, `v2` y `v3`, y `models/v3` debe tener `modelo.joblib`, `metrics.json` y `metadata.json`.

**Usarla en `/predict`:**

```powershell
$env:MODEL_VERSION = "v3"; uvicorn api:app --host 127.0.0.1 --port 8000
```

Para volver a la anterior, basta poner `MODEL_VERSION` en `v1` (o quitar la variable) y reiniciar.

### Errores

Todos los errores usan el mismo formato: `{"error": ..., "mensaje": ..., "detalles": [...]}`.

| Situación | Código | Ejemplo de mensaje |
|---|---|---|
| Falta una variable obligatoria | 422 | `Contract: campo obligatorio ausente` |
| Tipo incompatible | 422 | `tenure: debe ser un número entero (sin comillas)` |
| Valor fuera de rango | 422 | `SeniorCitizen: valor no permitido (se esperaba 0 o 1)` |
| Cuerpo que no es JSON | 422 | `cuerpo: no es un JSON válido` |
| Versión configurada no disponible | 503 | `El modelo de la versión 'v99' no está disponible. Versiones disponibles en 'models': v1, v2. Configure la variable de entorno MODEL_VERSION con una de ellas y reinicie la API.` |
| Ya hay un entrenamiento en curso (`/train`) | 409 | `Ya hay un entrenamiento en curso. Espere a que termine y vuelva a intentarlo.` |
| No existe el CSV de entrenamiento (`/train`) | 503 | `No se encontró el archivo CSV: data/... Coloque el CSV en 'data/...' (o monte la carpeta de datos en Docker) o configure la variable de entorno DATA_PATH.` |
| El CSV existe pero no se puede entrenar (`/train`) | 500 | `No se pudo entrenar` con el motivo |
| Variable de entorno inválida | 500 | `La variable de entorno SEED='abc' no es válida (se esperaba int).` |

Prueba del error de validación (`ejemplos/cliente_sin_contract.json`):

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@ejemplos/cliente_sin_contract.json"
```

```json
{"error": "Entrada inválida", "mensaje": "Contract: campo obligatorio ausente", "detalles": [{"campo": "Contract", "problema": "campo obligatorio ausente"}]}
```

## 6. Pendiente en los siguientes PR

- Dockerfile actualizado, pruebas con pytest y cliente de prueba.
- Comandos de Docker con montajes de datos y modelos.