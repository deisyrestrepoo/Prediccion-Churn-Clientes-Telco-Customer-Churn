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
| `feature/monitoreo-api` | API con el registro habilitado (una fila por intento, incluidos los rechazos), Dockerfile y montaje del registro | **Este PR** |
| `feature/monitoreo-reporte` | Sección «Reporte», política de reentrenamiento, versiones identificadas y muestra del registro | Pendiente |

## 2. Estructura

```
fase-4/
├── api.py                   # API de la Fase 3 + middleware que registra cada intento de /predict
├── registro.py              # Escribe una fila por solicitud en el registro CSV
├── monitoreo.py             # Lee el registro y muestra los dos indicadores
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
| `fecha_hora` | Fecha y hora local de la solicitud, formato ISO 8601 con precisión de segundos |
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


## 7. Pendiente en el siguiente PR

- Sección «Reporte» con las solicitudes enviadas, los dos indicadores obtenidos y una frase que los explique (ejecución real con la API en Docker).
- Política de reentrenamiento e identificadores de las dos versiones del modelo (con cómo volver a la anterior).
- Muestra del registro utilizado.
