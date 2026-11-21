# Imagen de la API (Fase 3): sirve /health, /predict y /train con uvicorn.
# Python con versión explícita (la misma del entorno local, 3.12.5); no se usa "latest".
FROM python:3.12.5-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1

WORKDIR /app

# Dependencias primero: Docker reutiliza esta capa si requirements.txt no cambia.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código. Los modelos y los datos NO van en la imagen: se montan al ejecutar (-v).
# El modelo no se entrena al construir la imagen; /train entrena dentro del contenedor.
COPY api.py pytest.ini ./
COPY src/ ./src/
COPY tests/ ./tests/

# Puerto de la API. Se publica al ejecutar con -p 8000:8000.
EXPOSE 8000

# Estado del contenedor (docker ps muestra "healthy" cuando /health responde).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# Sin ENTRYPOINT: el mismo contenedor puede arrancar la API (por defecto) o ejecutar pytest.
# 0.0.0.0 para que el puerto publicado sea accesible desde el equipo anfitrión.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]