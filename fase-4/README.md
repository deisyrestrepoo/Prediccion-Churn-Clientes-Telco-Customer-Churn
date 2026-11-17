# Fase 4 — Monitoreo de la API de predicción de churn

**Curso:** Modelos y Simulación de Sistemas I · Proyecto Integrador 2026-2 · Universidad de Antioquia
**Docente:** Andrés Parra
**Integrantes:** Deisy Tatiana Restrepo Ospina, Edgar Andrés Garzón Marín

## 1. Relación con la Fase 3

La Fase 4 comprueba **cómo se usa la API** de la Fase 3 y **qué versión del modelo predice**: cada llamada a `POST /predict` (aceptada o rechazada) deja una fila en un registro persistente, y un script calcula dos indicadores de operación. `fase-1/`, `fase-2/` y `fase-3/` se conservan sin cambios.

### Estado de la fase (por Pull Requests)

| PR / rama | Contenido | Estado |
|---|---|---|
| `feature/monitoreo-script` | Módulo de registro (`registro.py`), script `monitoreo.py` y sus pruebas | **Este PR** |
| `feature/monitoreo-api` | API con el registro habilitado (una fila por intento, incluidos los rechazos), Dockerfile y montaje del registro | Pendiente |
| `feature/monitoreo-reporte` | Sección «Reporte», política de reentrenamiento, versiones identificadas y muestra del registro | Pendiente |

## 2. Estructura

```
fase-4/
├── registro.py              # Escribe una fila por solicitud en el registro CSV
├── monitoreo.py             # Lee el registro y muestra los dos indicadores
├── tests/
│   └── test_monitoreo.py    # Pruebas del registro y del script (archivos temporales)
├── .gitignore               # Ignora logs/ (el registro real)
└── README.md
```

## 3. Formato del registro

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

## 4. Script de monitoreo

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

## 5. Pruebas

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

Las pruebas no necesitan la API, el modelo ni la red, y escriben solo en carpetas temporales.

## 6. Pendiente en los siguientes PR

- Registrar desde la API cada intento de `/predict`, incluidos los rechazados por validación, y verificar que cada intento genere una sola fila.
- Comando de inicio de la API con el registro habilitado y ubicación del registro persistente (montaje en Docker).
- Sección «Reporte» con las solicitudes enviadas, los dos indicadores obtenidos y una frase que los explique.
- Política de reentrenamiento e identificadores de las dos versiones del modelo (con cómo volver a la anterior).
- Muestra del registro utilizado.
