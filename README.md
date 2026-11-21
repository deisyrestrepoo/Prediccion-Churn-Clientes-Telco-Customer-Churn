# Predicción de Churn de Clientes — Telco Customer Churn

**Proyecto Integrador · Modelos y Simulación de Sistemas I · 2026-2 · Universidad de Antioquia**
**Docente:** Andrés Parra
**Integrantes:** Deisy Tatiana Restrepo Ospina, Edgar Andrés Garzón Marín

## El problema

Predecir si un cliente de una empresa de telecomunicaciones **abandonará el servicio (churn)** a partir de su contrato, los servicios que tiene contratados y sus cargos. Es una clasificación binaria sobre el dataset *Telco Customer Churn* (7.043 clientes, 21 columnas). El modelo es una Regresión Logística comparada con un modelo base; en el conjunto de prueba alcanza un ROC-AUC de 0.84 a 0.86 según la versión guardada (frente a 0.50 del modelo base).

## Fases del proyecto

Cada fase parte del trabajo de la anterior y la conserva sin reescribirla.

| Carpeta | Contenido |
|---|---|
| [`fase-1/`](fase-1/) | Modelo predictivo: notebook, modelo entrenado y README |
| [`fase-2/`](fase-2/) | Aplicación reutilizable: `train.py`, `predict.py`, modelos versionados, pruebas con pytest y Docker |
| [`fase-3/`](fase-3/) | API REST con FastAPI (`/health`, `/predict` y `/train`), Docker, pruebas con pytest y cliente de prueba |
| [`fase-4/`](fase-4/) | Monitoreo: registro de las solicitudes a `/predict`, métricas de operación y política de reentrenamiento |

Cada carpeta tiene su propio `README.md` con los comandos completos, la evidencia de ejecución y las limitaciones conocidas.

## Requisitos

- **Python 3.12** (referencia 3.12.5; no usar 3.14, porque scikit-learn 1.6.1 no tiene paquetes para esa versión). Las dependencias de cada fase están fijadas en su `requirements.txt`.
- **Docker Desktop**, para construir y ejecutar las imágenes de las fases 2 y 3.
- **Dataset:** el CSV completo no está en el repositorio. Se descarga de [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) y se deja en `fase-2/data/` y `fase-3/data/` con el nombre `WA_Fn-UseC_-Telco-Customer-Churn.csv` (esas carpetas están en el `.gitignore`).

## Inicio rápido: la API con Docker

Desde `fase-3/`, en PowerShell, con Docker Desktop abierto:

```powershell
docker build -t churn-api:1.0 .
docker run --rm -d --name churn-api -p 8000:8000 -v "${PWD}/models:/app/models" -v "${PWD}/data:/app/data:ro" churn-api:1.0
python cliente_prueba.py        # esperar a que `docker logs churn-api` muestre «Application startup complete»
docker stop churn-api
```

La documentación interactiva de la API queda en http://127.0.0.1:8000/docs. Los pasos completos están en [`fase-3/README.md`](fase-3/README.md).

## Flujo de trabajo (Git Flow simplificado)

- `main`: versiones estables de las entregas, identificadas con etiquetas (`entrega-1`, `entrega-2` y `entrega-3`, esta última para las Fases 3 y 4).
- `develop`: integración.
- `feature/*`: una rama por tarea, integrada mediante Pull Request hacia `develop` y revisada por el otro integrante del equipo.