# Fase 3 — API REST de predicción de churn

**Curso:** Modelos y Simulación de Sistemas I · Proyecto Integrador 2026-2 · Universidad de Antioquia
**Docente:** Andrés Parra
**Integrantes:** Deisy Tatiana Restrepo Ospina, Edgar Andrés Garzón Marín

## 1. Relación con la Fase 2

La Fase 3 expone por HTTP el modelo de la Fase 2 (clasificación de churn de clientes Telco): consultar una predicción, comprobar que el servicio está activo y entrenar una nueva versión. Se reutilizan la lógica de `fase-2/src/`, el esquema de entradas y el preprocesamiento guardado con cada versión. `fase-1/` y `fase-2/` se conservan sin cambios.

### Estado de la fase (por Pull Requests)

| PR / rama | Contenido | Estado |
|---|---|---|
| `feature/api-base` | Esqueleto de `fase-3/`, configuración y `GET /health` | **Este PR** |
| `feature/api-predict` | `POST /predict` con validación de entradas y errores 422 y 503 | Pendiente |
| `feature/api-train` | `POST /train` con versión nueva automática | Pendiente |
| `feature/api-docker-tests` | Dockerfile, pruebas con pytest, cliente de prueba y documentación completa | Pendiente |

## 2. Estructura

```
fase-3/
├── api.py             # Aplicación FastAPI (rutas)
├── requirements.txt   # Dependencias con versiones explícitas
├── src/               # Lógica reutilizable
│   ├── config.py, data.py, errors.py, persistence.py,
│   │   preprocessing.py, training.py   # Copia sin cambios de fase-2/src/
│   └── settings.py    # Nuevo: configuración de la API por variables de entorno
├── models/            # Versiones guardadas: v1/ y v2/ (copiadas de fase-2/models/)
└── README.md
```

**Origen del código.** Los archivos de `src/` (excepto `settings.py`) y las versiones `v1` y `v2` son una copia exacta de los de `fase-2`, para que `fase-3` se pueda construir y ejecutar por sí sola (incluido Docker). Se puede comprobar con `diff -r fase-2/src fase-3/src` y `diff -r fase-2/models/v1 fase-3/models/v1`: la única diferencia debe ser `settings.py`.

## 3. Configuración

La API se configura con variables de entorno, todas opcionales:

| Variable | Por defecto | Para qué sirve |
|---|---|---|
| `MODEL_VERSION` | `v1` | Versión del modelo que usa `/predict`. Para volver a la anterior basta cambiar este valor y reiniciar |
| `MODELS_DIR` | `models` | Carpeta con las versiones guardadas |
| `DATA_PATH` | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | CSV con datos etiquetados que usa `/train` |
| `TARGET` | `Churn` | Variable objetivo de `/train` |
| `SEED` | `42` | Semilla de `/train` |
| `TEST_SIZE` | `0.2` | Proporción de prueba de `/train` |
| `CV_FOLDS` | `5` | Folds de validación cruzada de `/train` |

## 4. Instalación y ejecución local

Python 3.12 (el mismo de la Fase 2). Desde la raíz del repositorio, con el entorno virtual activo (PowerShell):

```powershell
cd fase-3
pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8000
```

Para usar otra versión del modelo:

```powershell
$env:MODEL_VERSION = "v2"; uvicorn api:app --host 127.0.0.1 --port 8000
```

(En Bash: `MODEL_VERSION=v2 uvicorn api:app --host 127.0.0.1 --port 8000`.)

La documentación interactiva queda en http://127.0.0.1:8000/docs.

## 5. Rutas disponibles en este PR

### `GET /health`

Indica que la API está en ejecución y qué versión tiene configurada. Responde 200 aunque el modelo no esté disponible: `/health` describe el servicio, no la predicción.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
# Bash: curl http://127.0.0.1:8000/health
```

Respuesta:

```json
{"estado": "ok", "mensaje": "API en ejecución", "version_configurada": "v1", "modelo_disponible": true}
```

`modelo_disponible` es `false` si la versión configurada no existe en `MODELS_DIR` (por ejemplo, `MODEL_VERSION=v99`).

## 6. Pendiente en los siguientes PR

- `POST /predict`, `POST /train`, manejo de errores 422 y 503.
- Dockerfile actualizado, pruebas con pytest y cliente de prueba.
- Ejemplo de solicitud y respuesta, y comandos de Docker con montajes.