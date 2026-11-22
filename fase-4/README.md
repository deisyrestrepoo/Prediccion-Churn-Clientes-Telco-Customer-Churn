# Fase 4 — Monitoreo de la API de predicción de churn

**Curso:** Modelos y Simulación de Sistemas I · Proyecto Integrador 2026-2 · Universidad de Antioquia
**Docente:** Andrés Parra
**Integrantes:** Deisy Tatiana Restrepo Ospina, Edgar Andrés Garzón Marín

## 1. Relación con la Fase 3

La Fase 4 comprueba **cómo se usa la API** de la Fase 3 y **qué versión del modelo predice**: cada llamada a `POST /predict` (aceptada o rechazada) deja una fila en un registro persistente, y un script calcula dos indicadores de operación. `fase-1/`, `fase-2/` y `fase-3/` se conservan sin cambios.

### Estado de la fase (por Pull Requests)

| PR / rama | Contenido | Estado |
|---|---|---|
| `feature/monitoreo-script` | Módulo de registro (`registro.py`), script `monitoreo.py` y sus pruebas | Integrado en `develop` (PR #22) |
| `feature/monitoreo-api` | API con el registro habilitado (una fila por intento, incluidos los rechazos), Dockerfile y montaje del registro | Integrado en `develop` |
| `feature/monitoreo-reporte` | Sección «Reporte», política de reentrenamiento, versiones identificadas y muestra del registro | **Este PR** |

## 2. Estructura

```
fase-4/
├── api.py                   # API de la Fase 3 + middleware que registra cada intento de /predict
├── registro.py              # Escribe una fila por solicitud en el registro CSV
├── monitoreo.py             # Lee el registro y muestra los dos indicadores
├── muestra_registro.csv     # Registro usado en el reporte (3 solicitudes)
├── src/                     # Código reutilizado de la Fase 2 (esquema, datos, modelos, entrenamiento)
├── models/                  # Versiones guardadas (v1, v2)
├── ejemplos/                # Cliente de ejemplo en JSON para la API
├── cliente_prueba.py        # Cliente de prueba (igual que en la Fase 3)
├── tests/
│   ├── conftest.py          # Redirige el registro a una carpeta temporal en cada prueba
│   ├── test_api.py          # Pruebas de la API (heredadas de la Fase 3)
│   ├── test_api_registro.py # Pruebas del registro desde la API
│   └── test_monitoreo.py    # Pruebas del registro y del script (archivos temporales)
├── Dockerfile / .dockerignore / requirements.txt / pytest.ini
├── .gitignore               # Ignora logs/ (el registro real)
└── README.md
```

La Fase 4 incluye su **propia copia** de la API (`api.py`), igual que `fase-3/` es independiente de `fase-2/`: `fase-3/` queda intacta.

## 3. Arranque con el registro habilitado

El registro está **siempre habilitado**: cada llamada a `POST /predict` agrega una fila, sea aceptada (200) o rechazada (422, 503, 500...). `/health` y `/train` no generan filas. La ubicación se define con `PREDICTION_LOG` (por defecto `logs/predicciones.csv`) y la versión del modelo con `MODEL_VERSION` (por defecto `v1`).

**Local (PowerShell, desde `fase-4/`):**

```powershell
$env:MODEL_VERSION = "v1"
$env:PREDICTION_LOG = "logs\predicciones.csv"
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

**Docker (desde `fase-4/`, que es el contexto del build):**

```powershell
docker build -t churn-api:4.0 .
docker run --rm -p 8000:8000 `
  -e MODEL_VERSION=v1 `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/logs:/app/logs" `
  churn-api:4.0
```

Registro persistente: `fase-4/logs/predicciones.csv` en tu PC (la carpeta `logs/` se monta en `/app/logs`; la imagen fija `PREDICTION_LOG=/app/logs/predicciones.csv`). `logs/` está en `.gitignore`: el registro real crece con el uso y no se versiona; la muestra del reporte se entrega aparte.

**Cómo se garantiza una sola fila por intento.** Un middleware observa la respuesta final de `/predict`: toma el código HTTP que se devuelve (también el de las entradas que FastAPI rechaza antes de llegar a la función de predicción) y, si fue exitosa, la versión y la predicción que produjo la respuesta. Si el registro no se puede escribir, el error se anota en el log del servidor y la respuesta al cliente no cambia. Una excepción no controlada se registra como 500.

Decisión documentada: se registra **cualquier** llamada a la ruta `/predict`, también con un método no permitido (por ejemplo `GET`, que da 405), porque es un intento de uso de esa ruta y cuenta como fallido.

## 4. Formato del registro

Archivo CSV (por defecto `logs/predicciones.csv`) con un encabezado y una fila por solicitud a `/predict`:

```
fecha_hora,estado_http,version,prediccion
2026-11-17T10:00:00,200,v1,No churn
2026-11-17T10:01:00,200,v1,Churn
2026-11-17T10:02:00,422,,
```

| Campo | Contenido |
|---|---|
| `fecha_hora` | Fecha y hora de la solicitud en la hora del servidor que ejecuta la API (UTC en Docker), formato ISO 8601 con precisión de segundos |
| `estado_http` | Código HTTP devuelto (200, 422, 503, ...) |
| `version` | Versión del modelo que produjo la respuesta; vacía si no se usó ningún modelo |
| `prediccion` | Etiqueta predicha (`Churn` o `No churn`); vacía en un error |

Reglas de `registro.py`:

- El archivo se abre en modo **agregar**: el historial se conserva entre ejecuciones.
- El encabezado se escribe solo cuando el archivo se crea (o está vacío).
- La carpeta del registro se crea si no existe.
- Un candado evita que dos solicitudes simultáneas mezclen filas o repitan el encabezado.
- La ubicación se elige con la variable de entorno `PREDICTION_LOG` (por defecto `logs/predicciones.csv`). En Docker debe apuntar a una carpeta montada para que el registro persista.

## 5. Script de monitoreo

```powershell
cd fase-4
python monitoreo.py --log logs/predicciones.csv
```

Sin `--log` usa `PREDICTION_LOG` o, en su defecto, `logs/predicciones.csv`. Ejemplo de salida con dos solicitudes válidas y una rechazada:

```
Registro: logs/predicciones.csv
Total de solicitudes: 3
Solicitudes fallidas (HTTP 4xx o 5xx): 1
Porcentaje de fallos: 33,3 %
```

- **Total:** número de filas de datos del registro.
- **Fallidas:** filas con `estado_http` entre 400 y 599.
- **Porcentaje de fallos:** fallidas / total × 100, con un decimal.
- **Registro vacío** (archivo de 0 bytes o solo encabezado): `Total de solicitudes: 0` y `Porcentaje de fallos: sin solicitudes`.
- **Errores previsibles** (archivo inexistente, encabezado distinto del esperado, `estado_http` que no es un número): `Error: <mensaje>` y código de salida 1, sin traceback.

Este porcentaje describe el funcionamiento del servicio. Medir la calidad predictiva exigiría conocer los resultados reales de los clientes.

## 6. Pruebas

```powershell
cd fase-4
python -m pytest
```

| Prueba | Qué verifica |
|---|---|
| `test_registro_escribe_encabezado_una_vez_y_conserva_el_historial` | Encabezado único, modo agregar entre «ejecuciones», campos vacíos en los errores |
| `test_dos_validas_y_una_invalida_dan_3_solicitudes_y_33_3_por_ciento` | El caso de referencia de la especificación: 3 solicitudes y 33,3 % |
| `test_cuentan_como_fallidas_solo_4xx_y_5xx` | Frontera del rango: 399 no cuenta; 400 y 599 sí |
| `test_registro_vacio_muestra_total_0_y_sin_solicitudes` | Archivo vacío y archivo con solo encabezado |
| `test_registro_inexistente_o_invalido_da_error_claro_y_codigo_1` | Archivo inexistente, encabezado ajeno y código no numérico |
| `test_script_como_proceso_usa_la_variable_PREDICTION_LOG` | El script como proceso real, sin `--log` |

Pruebas del registro desde la API (`tests/test_api_registro.py`):

| Prueba | Qué verifica |
|---|---|
| `test_dos_validas_y_una_sin_campo_dejan_3_filas_y_33_3_por_ciento` | El ensayo de la especificación sobre un registro vacío: 200, 200, 422; exactamente 3 filas; versión y predicción vacías en el rechazo; el script da 3 y 33,3 % |
| `test_cada_tipo_de_rechazo_deja_una_sola_fila` | Tipo incompatible, JSON roto y cuerpo vacío: una fila (422) cada uno |
| `test_modelo_no_disponible_registra_503_sin_version_ni_prediccion` | Versión inexistente: fila 503 con versión y predicción vacías |
| `test_error_no_controlado_registra_500` | Excepción inesperada: fila 500 |
| `test_solo_se_registra_predict_y_el_historial_se_conserva` | `/health` no registra; encabezado único y modo agregar entre clientes |
| `test_el_cambio_de_version_se_ve_en_el_registro` | La columna `version` refleja el cambio de `MODEL_VERSION` |
| `test_metodo_no_permitido_en_predict_cuenta_como_intento_fallido` | `GET /predict` (405) deja una fila fallida (decisión documentada) |

Las pruebas usan un modelo pequeño con datos sintéticos, no necesitan el CSV de Kaggle ni la red, y escriben solo en carpetas temporales.


## 7. Reporte

Reporte de la ejecución real del 21 de noviembre de 2026, con la API en Docker (`churn-api:4.0`), `MODEL_VERSION=v1` y el registro **vacío** al empezar. Los comandos son los de la sección 3.

### 7.1 Solicitudes enviadas

Con `cliente_prueba.py`, cada una una sola vez, a `POST http://localhost:8000/predict`:

| # | Comando | Qué se envió | Respuesta |
|---|---|---|---|
| 1 | `python cliente_prueba.py` | Cliente `EJ-0001` (fila 0 de `ejemplos/ejemplo_prediccion.csv`), 19 variables | 200, `Churn`, versión `v1` |
| 2 | `python cliente_prueba.py --fila 1` | Cliente `EJ-0002`, 19 variables | 200, `No churn`, versión `v1` |
| 3 | `python cliente_prueba.py --quitar Contract` | Cliente `EJ-0001` **sin el campo `Contract`** | 422, el mensaje nombra el campo `Contract` |

Registro resultante (`logs/predicciones.csv`; copia en [`muestra_registro.csv`](muestra_registro.csv)). Las horas están en UTC, que es la hora del contenedor:

```
fecha_hora,estado_http,version,prediccion
2026-11-21T18:32:32,200,v1,Churn
2026-11-21T18:32:47,200,v1,No churn
2026-11-21T18:32:58,422,,
```

El log del contenedor (`docker logs churn-api4`) mostró tres líneas `POST /predict` (200, 200 y 422), una por fila del registro.

### 7.2 Indicadores

Salida de `python monitoreo.py` sobre ese registro:

```
Registro: logs\predicciones.csv
Total de solicitudes: 3
Solicitudes fallidas (HTTP 4xx o 5xx): 1
Porcentaje de fallos: 33,3 %
```

| Indicador | Valor |
|---|---|
| Total de solicitudes | **3** |
| Porcentaje de solicitudes fallidas | **33,3 %** (1 de 3) |

Coinciden con las filas guardadas: 3 filas de datos, una con código 422.

**Interpretación.** La API procesó dos registros con `v1` y rechazó el tercero con HTTP 422 porque faltaba el campo `Contract`; la tasa de fallos fue 1/3 (33,3 %), y el rechazo era el resultado esperado de la prueba de validación. Este porcentaje describe el funcionamiento del servicio, no la calidad de las predicciones.

*Nota sobre el ensayo.* En una primera ejecución se envió dos veces por error la solicitud sin `Contract`, lo que dio 4 solicitudes y 50,0 %; las 4 filas coincidían con las 4 líneas del log del contenedor. Esa corrida se descartó y se repitió sobre un registro vacío para obtener el caso de referencia que se reporta arriba.

### 7.3 Versiones del modelo

| | Versión activa | Versión anterior / alternativa |
|---|---|---|
| Identificador | **`v1`** | **`v2`** |
| Ubicación | `fase-4/models/v1/` | `fase-4/models/v2/` |
| Creación | 31-oct-2026 (Fase 2), semilla 42 | 31-oct-2026 (Fase 2), semilla 7 |
| ROC-AUC en su conjunto de prueba | 0.8416 | 0.8567 |

Ambas versiones guardan `modelo.joblib`, `metrics.json` y `metadata.json`, y también están en `fase-2/models/` y `fase-3/models/`. La API usa la que indica la variable `MODEL_VERSION` (por defecto `v1`, que fue la usada en esta ejecución; cada fila del registro lo confirma en la columna `version`).

Los ROC-AUC **no son comparables entre sí**: `v1` y `v2` se entrenaron con semillas distintas y, por tanto, con particiones de prueba distintas. Por eso `v2` no se activó, y la política de abajo exige comparar sobre los mismos casos.

**Cambiar de versión o volver a la anterior** (sin tocar código; reiniciar la API):

```powershell
# Local
$env:MODEL_VERSION = "v2"      # activar v2; "v1" para volver a la anterior
python -m uvicorn api:app --host 127.0.0.1 --port 8000

# Docker
docker run --rm -d --name churn-api4 -p 8000:8000 -e MODEL_VERSION=v2 -v "${PWD}/models:/app/models" -v "${PWD}/logs:/app/logs" churn-api:4.0
```

Si la versión no existe, `/predict` responde 503 con la lista de versiones disponibles, y esa solicitud queda registrada con el código 503. Una versión nueva creada con `POST /train` (`v3`, `v4`...) se activa de la misma forma.

### 7.4 Política de reentrenamiento

Al reunir 500 clientes nuevos con su resultado real (si abandonaron o no en el periodo siguiente), o si el ROC-AUC de la versión activa sobre esos casos baja de 0.80, el equipo revisará el modelo: un integrante ejecutará `POST /train` para generar una versión candidata, sin tocar la activa. Comparará la candidata y la versión activa con el ROC-AUC (la métrica principal de la Fase 2) sobre los mismos casos reservados y activará la candidata solo si supera a la activa por al menos 0.01, cambiando `MODEL_VERSION` y reiniciando la API. Si la nueva versión empeora en producción, se vuelve a la anterior con la misma variable; el porcentaje de fallos del registro no sirve para esta decisión, porque mide el servicio y no la calidad predictiva.

### 7.5 Entregables de la fase

| Entregable | Archivo |
|---|---|
| Código de registro | [`registro.py`](registro.py) y el middleware de [`api.py`](api.py) |
| Script de monitoreo | [`monitoreo.py`](monitoreo.py) |
| Muestra del registro usado en el reporte | [`muestra_registro.csv`](muestra_registro.csv) |
| Pruebas (22 con pytest) | [`tests/`](tests/) |

## 8. Limitaciones

- El registro es un CSV local sin rotación: crece con el uso y no hay tableros ni alertas (son opcionales en la especificación).
- Las horas se guardan en la hora del servidor que ejecuta la API (UTC en Docker).
- Se registra cualquier llamada a la ruta `/predict`, incluidas las de método no permitido (405), que cuentan como fallidas.
- La política de reentrenamiento está documentada, no automatizada. Los umbrales (500 casos, ROC-AUC 0.80 y mejora de 0.01) son una decisión del equipo para este proyecto y se pueden ajustar.
- No se mide la calidad predictiva en producción: el dataset no trae resultados reales posteriores a la predicción.
