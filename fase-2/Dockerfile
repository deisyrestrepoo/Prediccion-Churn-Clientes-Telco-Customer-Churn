# Imagen de la Fase 2: train.py y predict.py se ejecutan desde la misma imagen.
# Python con versión explícita (la misma del entorno local, 3.12.5); no se usa "latest".
FROM python:3.12.5-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1

WORKDIR /app

# Dependencias primero: Docker reutiliza esta capa si requirements.txt no cambia.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código. Los datos y los modelos no van en la imagen: se montan al ejecutar (-v).
# El modelo se entrena al ejecutar el contenedor.
COPY src/ ./src/
COPY tests/ ./tests/
COPY train.py predict.py ./

# Sin ENTRYPOINT: el mismo contenedor puede ejecutar train.py, predict.py o pytest.
# Por defecto solo muestra la ayuda de train.py.
CMD ["python", "train.py", "--help"]
