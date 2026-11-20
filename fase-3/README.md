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
| `feature/api-train` | `POST /train` con versión nueva automática | Integrado en `develop` |
| `feature/api-docker-tests` | Dockerfile, pruebas con pytest, cliente de prueba y documentación completa | **Este PR** |

## 2. Estructura

```
fase-3/
├── api.py             # Aplicación FastAPI (rutas y manejo de errores)
├── requirements.txt   # Dependencias con versiones explícitas (incluye pytest)
├── Dockerfile         # Imagen de la API (uvicorn en 0.0.0.0:8000)
├── .dockerignore      # Excluye entornos, cachés, credenciales, datos y modelos de la imagen
├── pytest.ini         # Configuración de pytest
├── cliente_prueba.py  # Cliente de prueba: envía una fila del CSV de ejemplo a /predict
├── src/
│   ├── config.py, data.py, errors.py, persistence.py,
│   │   preprocessing.py, training.py   # Copia sin cambios de fase-2/src/
│   ├── settings.py    # Configuración por variables de entorno
│   ├── schemas.py     # Entradas y salidas de /predict y /train
│   └── servicio.py    # Carga de la versión, predicción y entrenamiento de versiones nuevas
├── tests/
│   └── test_api.py    # Pruebas con pytest (8 pruebas)
├── models/            # Versiones guardadas: v1/ y v2/ (copiadas de fase-2/models/)
├── ejemplos/          # JSON y CSV de ejemplo para probar la API
├── data/              # CSV completo del dataset (no se sube a Git; ver sección 3)
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

## 6. Docker

La imagen contiene solo el código de la API (`api.py`, `src/`, `tests/`) y las dependencias fijadas de `requirements.txt`, sobre `python:3.12.5-slim`. **No** contiene datos, modelos ni credenciales (ver `.dockerignore`) y **no entrena durante el build**: `/train` entrena dentro del contenedor cuando se le pide. Los modelos y los datos se montan desde el equipo anfitrión, así que las versiones que cree `/train` persisten aunque se elimine el contenedor.

Los comandos se ejecutan en **PowerShell**, desde `fase-3/` (con Docker Desktop abierto):

**Construir la imagen**

```powershell
docker build -t churn-api:1.0 .
```

**Iniciar la API**

```powershell
docker run --rm -d --name churn-api -p 8000:8000 -v "${PWD}/models:/app/models" -v "${PWD}/data:/app/data:ro" churn-api:1.0
```

| Opción | Para qué sirve |
|---|---|
| `-p 8000:8000` | Publica el puerto 8000 de la API (si está ocupado, usar por ejemplo `-p 8001:8000`) |
| `-v "${PWD}/models:/app/models"` | Monta las versiones del modelo. Es de escritura: `/train` guarda aquí las versiones nuevas |
| `-v "${PWD}/data:/app/data:ro"` | Monta el CSV de entrenamiento en solo lectura (solo lo usa `/train`) |
| `-e MODEL_VERSION=v2` | (Opcional) Usa otra versión del modelo; por defecto `v1` |

**Comprobar que funciona**

```powershell
docker ps                                       # STATUS pasa a "healthy" en unos segundos
Invoke-RestMethod http://127.0.0.1:8000/health
python cliente_prueba.py                        # envía un cliente del CSV de ejemplo a /predict
docker logs churn-api                           # solicitudes recibidas
```

Nota: en Windows PowerShell 5, `Invoke-RestMethod` puede mostrar mal los acentos y no imprime el JSON completo; para ver la respuesta tal cual se usa `cliente_prueba.py` o `curl.exe`. Al iniciar el contenedor, la API tarda unos segundos en estar lista: conviene esperar a que `docker logs churn-api` muestre `Application startup complete` antes de consultar.

**Entrenar una versión nueva dentro del contenedor** (necesita el CSV en `data/`):

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/train
dir models                                      # aparece la versión nueva (v3) en la carpeta local
```

**Detener**

```powershell
docker stop churn-api
```

**Pruebas dentro del contenedor**

```powershell
docker run --rm churn-api:1.0 python -m pytest -q
```

## 7. Pruebas con pytest

Las pruebas usan un modelo pequeño entrenado al vuelo con datos sintéticos que tienen el esquema real: no necesitan el CSV de Kaggle, ni la red, ni las versiones de `models/`, y escriben solo en carpetas temporales. Desde `fase-3/`, con el entorno virtual activo:

```powershell
python -m pytest -v
```

| Prueba | Qué verifica |
|---|---|
| `test_health_responde_200` | **(mínima 1)** `/health` responde 200 e informa el estado y la versión configurada |
| `test_entrada_valida_devuelve_prediccion_y_version` | **(mínima 2)** Una entrada válida devuelve predicción y versión, y la probabilidad coincide con la del artefacto guardado |
| `test_entrada_sin_campo_obligatorio_se_rechaza` | **(mínima 3)** Sin una variable obligatoria: 422 y el mensaje nombra el campo |
| `test_tipo_incompatible_se_rechaza_nombrando_el_campo` | Un tipo incompatible (`tenure` como texto) da 422 con el nombre del campo |
| `test_valor_nulo_permitido_usa_la_regla_guardada` | Una cuenta sin primera factura (`TotalCharges` `null`) se predice con la regla de la Fase 2 |
| `test_categoria_no_vista_se_predice_con_aviso` | Una categoría no vista da 200 con un aviso |
| `test_modelo_no_disponible_devuelve_503_y_lista_versiones` | Versión inexistente: 503 con las versiones disponibles; `/health` sigue en 200 |
| `test_train_guarda_versiones_nuevas_sin_tocar_las_anteriores` | `/train` crea `v1` y luego `v2`, y el modelo anterior queda idéntico (misma huella SHA-256) |

Los `DeprecationWarning` de scipy que pueden aparecer con estas versiones no afectan los resultados.

## 8. Cliente de prueba

`cliente_prueba.py` (solo biblioteca estándar) toma una fila de `ejemplos/ejemplo_prediccion.csv` (el CSV de ejemplo de la Fase 2), la convierte en JSON —números como números, categorías como texto y celdas vacías como `null`— y la envía a `POST /predict`. Con la API en marcha, desde `fase-3/`:

```powershell
python cliente_prueba.py                          # fila 0 (cliente EJ-0001)
python cliente_prueba.py --fila 6                 # EJ-0007: tenure 0 y TotalCharges vacío -> null
python cliente_prueba.py --quitar Contract        # quita un campo: debe responder 422
python cliente_prueba.py --url http://127.0.0.1:8001
```

Termina con código 0 si la API respondió lo esperado (200, o 422 con `--quitar`) y con 1 en otro caso, incluso si no puede conectarse. Salida con la fila 0:

```
GET /health -> HTTP 200: {"estado": "ok", "mensaje": "API en ejecución", "version_configurada": "v1", "modelo_disponible": true}

POST /predict con el cliente EJ-0001:
{ "gender": "Female", "SeniorCitizen": 0, ..., "MonthlyCharges": 29.85, "TotalCharges": 29.85 }

-> HTTP 200:
{ "prediccion": "Churn", "clase": 1, "probabilidad_churn": 0.8057, "version": "v1", "avisos": [] }
```

El mismo cliente en `curl` (equivale al ejemplo de la sección 5):

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@ejemplos/cliente_valido.json"
```

## 9. Evidencia de ejecución

Ejecutada en Windows (PowerShell, Docker Desktop 4.97.0, Python 3.12.10) el 19 de noviembre de 2026, desde `fase-3/`.

| Verificación | Resultado |
|---|---|
| `python -m pytest -v` en el equipo local | `8 passed` en 17.05 s |
| `docker build -t churn-api:1.0 .` | Imagen creada. El primer build tardó 1266 s por la descarga de las dependencias; los siguientes reutilizan la caché |
| `docker run --rm churn-api:1.0 ls -a /app` | Solo `api.py`, `pytest.ini`, `requirements.txt`, `src` y `tests`: la imagen no contiene datos, modelos ni credenciales |
| Iniciar el contenedor (`docker run --rm -d --name churn-api -p 8000:8000 -v ...`) | `docker ps` muestra `Up About a minute (healthy)`; los registros muestran `Application startup complete` y `Uvicorn running on http://0.0.0.0:8000` |
| `GET /health` desde el equipo anfitrión | HTTP 200: `{"estado": "ok", "mensaje": "API en ejecución", "version_configurada": "v1", "modelo_disponible": true}` |
| `python cliente_prueba.py` | HTTP 200: `{"prediccion": "Churn", "clase": 1, "probabilidad_churn": 0.8057, "version": "v1", "avisos": []}` |
| `POST /train` dentro del contenedor | Respuesta: `Versión v3 guardada; las anteriores se conservan. /predict sigue usando 'v1'...` |
| Persistencia | Con el contenedor ya eliminado (`docker ps -a` sin filas), `models/v3/` existe en la carpeta local con `modelo.joblib`, `metrics.json` y `metadata.json`; su ROC-AUC es 0.8416, igual que `v1`; `v1` y `v2` siguen en su lugar |
| `docker run --rm churn-api:1.0 python -m pytest -q` | `8 passed` en 3.56 s |

Solicitud y respuesta reales del cliente de prueba (cliente `EJ-0001` del CSV de ejemplo):

```json
{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1,
 "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
 "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
 "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
 "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
 "MonthlyCharges": 29.85, "TotalCharges": 29.85}
```

```json
{"prediccion": "Churn", "clase": 1, "probabilidad_churn": 0.8057, "version": "v1", "avisos": []}
```

La versión `v3` creada para esta prueba se eliminó después para no incluirla en el repositorio; las versiones de la entrega siguen siendo `v1` y `v2`.