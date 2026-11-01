# Fase 2 — Scripts y Docker: Predicción de Churn de clientes Telco

**Curso:** Modelos y Simulación de Sistemas I · Proyecto Integrador 2026-2 · Universidad de Antioquia
**Docente:** Andrés Parra
**Integrantes:** Deisy Tatiana Restrepo Ospina, Edgar Andrés Garzón Marín
**Entrega 2:** 31 de octubre de 2026



## 1. Problema y relación con la Fase 1

El problema no cambia: **predecir si un cliente de una empresa de telecomunicaciones abandonará el servicio (churn)** a partir de su contrato, servicios contratados y cargos. Es una clasificación binaria sobre el dataset *Telco Customer Churn* (7.043 clientes, 21 columnas).

En la Fase 1 todo el trabajo vivía en `fase-1/notebook.ipynb`. La Fase 2 convierte ese notebook en una **aplicación reutilizable**: el modelo se entrena con un comando, queda guardado en una versión identificable y puede reutilizarse sin abrir el notebook. Se conservan el problema, los datos y el enfoque de modelado (Regresión Logística frente a un modelo base); `fase-1/` no se modificó.

### Estado de la fase (por Pull Requests)

| PR / rama | Contenido | Estado |
|---|---|---|
| (estructura inicial) | Carpetas y archivos vacíos de `fase-2/`, `.gitignore`, `requirements.txt` | Integrado en `develop` |
| `feature/train-script` | Lógica compartida en `src/` + `train.py` + este README | Integrado en `develop` (PR #13) |
| `feature/reproducibilidad-versiones` | Regeneración de `v1` y `v2` con el entorno de `requirements.txt` | Integrado a develop |
| `feature/predict-script` | `predict.py` + CSV de ejemplo | **Este PR** |
| `feature/docker-tests` | `Dockerfile`, `.dockerignore` y pruebas con pytest | Pendiente |

Las secciones de predicción, Docker y pruebas de este README se completan en sus PR correspondientes (ver sección 10).

## 2. Qué se hizo en este PR

```
fase-2/
├── train.py              # Punto de entrada: entrena y guarda una versión del modelo
├── requirements.txt      # Dependencias con versiones explícitas
├── src/                  # Lógica reutilizable (también la usará predict.py)
│   ├── config.py         # Esquema del problema: columnas, etiquetas, nombres de archivos
│   ├── data.py           # Lectura del CSV, validación de columnas y de tipos
│   ├── preprocessing.py  # Construcción del Pipeline (preprocesamiento + modelo) y del modelo base
│   ├── training.py       # Partición, validación cruzada, entrenamiento y métricas
│   ├── persistence.py    # Versionamiento, metadatos, guardado y carga del artefacto
│   └── errors.py         # Errores previsibles con mensajes claros
├── models/               # Versiones guardadas: v1/, v2/ ...
├── data/                 # Aquí va WA_Fn-UseC_-Telco-Customer-Churn.csv (no se sube a Git; ver sección 5)
├── outputs/              # Resultados generados (ignorado por Git)
└── tests/                # Pruebas (PR feature/docker-tests)
```

**Por qué esta organización.** La especificación pide que `train.py` y `predict.py` sean solo puntos de entrada y que la lógica esté en funciones reutilizables, sin copiar las celdas del notebook. Cada módulo de `src/` tiene una sola responsabilidad (datos, preprocesamiento, entrenamiento, persistencia), importar un módulo **no** entrena ni lee archivos, y `train.py` solo interpreta argumentos, llama a esas funciones y muestra el resumen. Esto también permite reutilizar las mismas funciones en la API REST de la fase siguiente.

## 3. Correcciones y cambios respecto a la Fase 1

La especificación exige documentar los cambios que afecten el preprocesamiento, la validación o el modelo.

| Tema | Fase 1 (notebook) | Fase 2 | Motivo |
|---|---|---|---|
| Valores faltantes | Se eliminaban las 11 filas con `TotalCharges` vacío (clientes nuevos, `tenure = 0`) sobre todo el dataset | No se elimina ninguna fila: se imputan con la **mediana calculada solo con el entrenamiento**, dentro del Pipeline | Evitar perder observaciones y poder predecir sobre clientes con nulos sin descartarlos |
| Orden de las transformaciones | Mapeo de binarias y `get_dummies` sobre todo el dataset **antes** de separar | La partición se hace **primero** y todas las transformaciones se ajustan después, solo con entrenamiento | Evitar fuga de información |
| Codificación | `get_dummies(drop_first=True)` | `OneHotEncoder(handle_unknown="ignore")` dentro del Pipeline | Una categoría no vista en el entrenamiento no rompe la predicción: se codifica con ceros |
| Preprocesamiento guardado | Solo `modelo.joblib` del estimador | Se guarda el **Pipeline completo** (imputación + escalamiento + codificación + modelo) | Una predicción futura aplica exactamente las reglas aprendidas |
| Validación | Partición entrenamiento/prueba | Misma partición **más** validación cruzada estratificada de 5 folds sobre el entrenamiento (el Pipeline se reajusta en cada fold) | Estimar la variabilidad de la métrica sin usar el conjunto de prueba |
| Versionamiento | Un único `modelo.joblib` | Carpetas `models/<version>/` con modelo, métricas y metadatos; una versión existente nunca se sobrescribe | Requisito de la especificación (sección 10) |

## 4. Entorno

- **Python:** 3.12 (probado con 3.12.5; verificar con `python --version`; debe ser el mismo en local y en Docker). No usar Python 3.14: scikit-learn 1.6.1 no tiene paquetes para esa versión. Las versiones de las librerías están fijadas en `requirements.txt`.
- Terminal usada: **PowerShell** (Windows). En Bash cambia solo la activación del entorno virtual.

Desde la raíz del repositorio:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Bash: source .venv/Scripts/activate
cd fase-2
pip install -r requirements.txt
```

## 5. Datos

El CSV completo **no se sube al repositorio** (está en `.gitignore`). Para reproducir:

1. Descargar *Telco Customer Churn* desde Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   (o con la CLI de Kaggle: `kaggle datasets download -d blastchar/telco-customer-churn --unzip -p data`; las credenciales de Kaggle son personales y nunca se incluyen en el repositorio).
2. Dejar el archivo en `fase-2/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

Variable objetivo: `Churn` (`Yes` = el cliente se fue → clase 1; `No` → clase 0). `customerID` es un identificador y no se usa como predictora.

## 6. Entrenamiento: `train.py`

Comando (desde `fase-2/`, en una sola línea):

```powershell
python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn --output-dir models --version v1 --seed 42
```

Consultar la ayuda: `python train.py --help`.

| Argumento | Obligatorio | Por defecto | Descripción |
|---|---|---|---|
| `--data` | Sí | — | Ruta del CSV con datos etiquetados |
| `--version` | Sí | — | Identificador único de la versión (`v1`, `v2`, ...). Solo letras, números, `.`, `_`, `-` |
| `--target` | No | `Churn` | Nombre de la variable objetivo |
| `--output-dir` | No | `models` | Carpeta donde se guardan las versiones |
| `--seed` | No | `42` | Semilla aleatoria |
| `--test-size` | No | `0.2` | Proporción del conjunto de prueba |
| `--cv-folds` | No | `5` | Folds de validación cruzada sobre entrenamiento (`0` la omite) |

### Qué hace, paso a paso

1. **Valida** el identificador de versión y comprueba que no exista ya (si existe, falla **antes** de entrenar).
2. **Lee y valida** el CSV: que exista, que tenga la variable objetivo y las 19 columnas predictoras, que `Churn` solo contenga `Yes`/`No` y que las columnas numéricas no tengan texto no numérico. Un `TotalCharges` vacío es un faltante legítimo, no un error.
3. **Separa** entrenamiento y prueba (80/20, estratificado por `Churn`, con la semilla indicada) **antes** de ajustar cualquier transformación. No hay grupos naturales que respetar: cada fila es un cliente distinto.
4. **Valida con validación cruzada** (5 folds) solo sobre el entrenamiento.
5. **Entrena** el Pipeline completo con el entrenamiento y **evalúa** sobre el conjunto de prueba, que no se usó para ajustar nada.
6. **Evalúa el modelo base** (predice siempre la clase más frecuente) con la misma partición y las mismas métricas, para conservar la comparación de la Fase 1.
7. **Guarda** la versión y muestra un resumen con las rutas generadas.

### Preprocesamiento (aprendido solo con el entrenamiento)

| Grupo | Columnas | Tratamiento |
|---|---|---|
| Numéricas | `tenure`, `MonthlyCharges`, `TotalCharges` | Imputación por mediana + `StandardScaler` |
| Binaria numérica | `SeniorCitizen` (0/1) | Imputación por valor más frecuente |
| Categóricas (15) | `gender`, `Partner`, `Contract`, `PaymentMethod`, etc. | Imputación por categoría más frecuente + One-Hot (`handle_unknown="ignore"`) |

Modelo: **Regresión Logística** (`class_weight="balanced"`, `max_iter=1000`), el mismo enfoque de la Fase 1. La clase `Churn` es minoritaria (~26 %), por eso se pondera.

### Archivos que genera cada versión

```
models/
├── v1/
│   ├── modelo.joblib   # Pipeline completo + esquema de columnas + significado de etiquetas
│   ├── metrics.json    # Métricas del modelo, del modelo base y de la validación cruzada
│   └── metadata.json   # Datos de la ejecución (ver abajo)
└── v2/ ...
```

**`metadata.json`** incluye: versión y fecha de creación, tipo de problema, variable objetivo y algoritmo, parámetros del modelo y semilla, variables de entrada con sus tipos esperados, datos usados (fuente, archivo, `sha256` y cantidad de observaciones), estrategia de partición y tamaños, métrica principal, versiones de Python y scikit-learn, referencia a `requirements.txt` y el **hash del commit** de Git del código usado.

### Reglas de versionamiento

- Cada ejecución conservada tiene un identificador único; la versión es una carpeta.
- **Nunca se sobrescribe**: repetir un identificador termina con un mensaje claro y código de salida `1`, y no modifica los archivos existentes.
- El código se versiona en Git y los artefactos del modelo en `models/`; son cosas distintas.

### Manejo de errores

Los errores previsibles (archivo inexistente, columna faltante, objetivo con valores inválidos, texto en una columna numérica, versión repetida o con caracteres no válidos) imprimen `Error: <mensaje>` y terminan con código de salida distinto de cero, sin dejar una versión a medias.

## Predicción: `predict.py`

`predict.py` carga una versión guardada del modelo y predice sobre un CSV nuevo. **No reentrena ni reajusta nada**: usa el Pipeline guardado (imputación, escalamiento y codificación aprendidos en el entrenamiento).

Comando (desde `fase-2/`, en una sola línea):

```powershell
python predict.py --version v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv --probabilidades
```

Equivalente indicando la ruta del modelo: `python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv`. Consultar la ayuda: `python predict.py --help`.

| Argumento | Obligatorio | Descripción |
|---|---|---|
| `--version` o `--model` | Uno de los dos | `--version v1` busca `models/v1` (carpeta cambiable con `--models-dir`); `--model` acepta `modelo.joblib` o la carpeta de la versión |
| `--input` | Sí | CSV con los clientes a predecir (sin la variable objetivo) |
| `--output` | Sí | CSV de salida con las predicciones |
| `--probabilidades` | No | Agrega la columna `probabilidad_churn` |

### CSV de entrada

Debe tener las 19 columnas predictoras del dataset (en cualquier orden): `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges` y `TotalCharges`. Numéricas: `tenure`, `MonthlyCharges`, `TotalCharges` y `SeniorCitizen` (0/1); el resto son texto. Se incluye un ejemplo de 8 clientes inventados en `data/ejemplo_prediccion.csv`.

- **Columnas adicionales** (`customerID`, `Churn`, etc.): se ignoran para predecir. Si viene `customerID`, se conserva en la salida.
- **Valores nulos:** una celda vacía no es un error. Los numéricos se imputan con la **mediana del entrenamiento** y los categóricos con la categoría más frecuente del entrenamiento. Una **columna ausente** sí es un error.
- **Categorías no vistas** en el entrenamiento: se informa con un `Aviso` y la fila se predice igual (todas las columnas codificadas de esa variable quedan en 0).
- **Texto en una columna numérica:** error; el archivo no se procesa.

### Salida

Una fila por cada cliente de entrada, **en el mismo orden**; nunca se descarta una fila en silencio.

| Columna | Significado |
|---|---|
| `customerID` | Identificador (solo si venía en la entrada) |
| `prediccion` | `1` = el cliente probablemente abandonará (Churn); `0` = se quedará (No churn) |
| `prediccion_etiqueta` | `Churn` o `No churn` |
| `probabilidad_churn` | Probabilidad de la clase 1 (solo con `--probabilidades`); el umbral de decisión es 0.5 |

### Errores previsibles

Se imprime `Error: <mensaje>` por la salida de error, el código de salida es `1` y **no se genera el archivo de salida**.

| Situación | Mensaje |
|---|---|
| Falta una columna | `Faltan columnas requeridas: tenure` |
| Texto en columna numérica | `La columna 'MonthlyCharges' debe ser numérica, pero tiene valores no numéricos` |
| CSV inexistente o vacío | `No se encontró el archivo CSV: ...` |
| Versión o modelo inexistente | `No se encontró el modelo: ...` |
| Ni `--model` ni `--version` (o ambos) | `Indique exactamente una opción: --model o --version.` |

Salida real con la versión `v1`:

```
Modelo versión v1: 8 predicciones (4 con churn, 4 sin churn).
Archivo generado: outputs\predicciones.csv

customerID,prediccion,prediccion_etiqueta,probabilidad_churn
EJ-0001,1,Churn,0.8057
EJ-0002,0,No churn,0.1176
EJ-0003,1,Churn,0.8758
EJ-0004,0,No churn,0.0644
EJ-0005,1,Churn,0.9087
EJ-0006,0,No churn,0.0078
EJ-0007,0,No churn,0.1451
EJ-0008,1,Churn,0.7939
```

## 7. Evidencia de ejecución

Salida real de la ejecución en PowerShell (Windows):

```
PS ...\fase-2> python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn --output-dir models --version v1 --seed 42

=== Entrenamiento finalizado (versión v1) ===
Entrenamiento: 5634 filas | Prueba: 1409 filas

Métrica      Modelo base  Reg. Logística   (conjunto de prueba)
roc_auc           0.5000          0.8416
recall            0.0000          0.7834
precision         0.0000          0.5043
f1                0.0000          0.6136
accuracy          0.7346          0.7381

Validación cruzada (5 folds, solo entrenamiento): ROC-AUC = 0.8459 ± 0.0124

Archivos generados:
  modelo   : models\v1\modelo.joblib
  metricas : models\v1\metrics.json
  metadata : models\v1\metadata.json
```

Los mismos comandos con la misma semilla dan **resultados idénticos en Windows**, lo que confirma la reproducibilidad.

## 8. Resultados

Métrica principal: **ROC-AUC** sobre el conjunto de prueba (1.409 clientes). Se reportan también recall, precision, F1 y accuracy.

| Versión | Semilla | ROC-AUC | Recall | Precision | F1 | Accuracy | ROC-AUC en CV (media ± desv.) |
|---|---|---|---|---|---|---|---|
| `v1` | 42 | 0.8416 | 0.7834 | 0.5043 | 0.6136 | 0.7381 | 0.8459 ± 0.0124 |
| `v2` | 7 | 0.8567 | 0.8235 | 0.5394 | 0.6519 | 0.7665 | 0.8414 ± 0.0041 |
| Modelo base (ambas) | — | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.7346 | — |

> **Reproducibilidad del entorno.** `v1` y `v2` se regeneraron con Python 3.12.5 y scikit-learn 1.6.1 (las versiones de `requirements.txt`), y `metadata.json` registra ese entorno. Los resultados de esta tabla no cambiaron respecto a la primera ejecución. Las versiones se pueden cargar y usar con `predict` y `predict_proba` en ese entorno.

**Comparación con el modelo base.** El modelo base obtiene una accuracy de 0.7346 solo por predecir siempre "no se va", pero no detecta ningún cliente que abandona (recall 0). La Regresión Logística detecta entre el 78 % y el 82 % de los clientes que se van. Por eso la accuracy no es una buena métrica aquí y se usa ROC-AUC y recall.

**Comparación con la Fase 1.** El notebook reportaba ROC-AUC 0.8353 y recall 0.7941. Los valores actuales difieren ligeramente porque ya no se eliminan las 11 filas con `TotalCharges` vacío y porque el preprocesamiento ahora se ajusta después de la partición. La especificación no exige mejorar la métrica de la Fase 1.

**`v1` y `v2` solo difieren en la semilla** (que cambia la partición). La diferencia de 0.015 en ROC-AUC muestra cuánto depende el resultado de qué clientes caen en la prueba, y por eso se reporta también la validación cruzada.

### Limitaciones conocidas

- La precisión es ~0.50–0.54: aproximadamente la mitad de los clientes marcados como "se va" en realidad se quedan. Es el costo de ponderar la clase minoritaria para lograr un recall alto.
- Se usa el umbral de decisión por defecto (0.5); no se optimizó.
- La Regresión Logística es determinista (el solver lbfgs no es aleatorio): la semilla controla la partición y la validación cruzada, no el ajuste del modelo. Con otras versiones de scikit-learn pueden aparecer diferencias numéricas mínimas.
- `TotalCharges` vacío corresponde a clientes con `tenure = 0`; imputarlo con la mediana es una simplificación (su valor lógico sería 0).
- El `commit` en `metadata.json` solo se registra si Git está disponible; dentro de Docker queda como "no disponible".

## 9. Cómo se revisó este PR

1. La estructura coincide con la sección 13 de la especificación.
2. `train.py` funciona sin ejecutar el notebook y permite consultar `--help`.
3. La partición ocurre antes de ajustar transformaciones, y el Pipeline se guarda completo.
4. Se guardan métricas del modelo y del modelo base, y metadatos de cada versión.
5. Existen dos versiones (`v1`, `v2`) y repetir un identificador no sobrescribe.
6. Los resultados se reprodujeron con los mismos números en otra máquina y otro sistema operativo.

## 10. Pendiente en los siguientes PR

- **`feature/predict-script`:** `predict.py`, CSV de ejemplo `data/ejemplo_prediccion.csv`, esquema del CSV de predicción, validación de columnas y tipos, y significado de la salida.
- **`feature/docker-tests`:** `Dockerfile`, `.dockerignore`, comandos de construcción y ejecución con montajes, y las cinco pruebas mínimas con pytest (`python -m pytest` desde `fase-2/`).
- **Cierre:** integrar `develop` en `main` y etiquetar la entrega (`entrega-2`); el commit o etiqueta se registrará aquí.

## 11. Fuentes

- Dataset: *Telco Customer Churn* (IBM sample data), Kaggle — https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Documentación de scikit-learn: `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, `LogisticRegression`, `StratifiedKFold`.
- Especificación técnica de la Fase 2 (docente Andrés Parra) y notebook de la Fase 1 del equipo.