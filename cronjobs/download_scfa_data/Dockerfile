FROM --platform=linux/amd64 tiangolo/uvicorn-gunicorn:python3.11-slim

RUN apt-get update && apt-get install -y netcat-openbsd

WORKDIR /cron/

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

ENV PYTHONPATH=/cron \
    DB_NAME=postgres \
    DB_USER=postgres \
    DB_PASSWORD=changeme \
    DB_HOST=timescaledb.default.svc.cluster.local \
    DB_PORT=5432 \
    MONGO_DB_USER=finsc \
    MONGO_DB_PASSWORD=finsc \
    MONGO_DB_HOST=finsc-mongodb-svc.default.svc.cluster.local \
    MONGO_DB_PORT=27017 \
    PROJECT_NAME=FinSC \
    DEBUG_LOGS=false \
    ECHO_SQL=true \
    VERSION=0.1 \

CMD ["python3.11", "cron_scfa_data.py"]