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
    PROJECT_NAME=FinSC \
    DEBUG_LOGS=false \
    ECHO_SQL=true \
    AUTH0_DOMAIN=finsc-dev-auth.jp.auth0.com \
    AUTH0_ISSUER=https://finsc-dev-auth.jp.auth0.com/ \
    AUTH0_ALGORITHMS=RS256 \
    AUTH0_API_AUDIENCE=http://finsc.localhost/ \
    DNSE_USERNAME=y.huynhngoc99@gmail.com \
    DNSE_PASSWORD=finsc6868@ \
    DNSE_HOST=datafeed-lts.dnse.com.vn \
    DATAFEED_BUCKET_NAME=finsc-datafeed \
    VERSION=0.1 \
    DESCRIPTION="FinSC API Trading"

CMD ["python3.11", "download_stockprice_from_s3.py"]