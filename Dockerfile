FROM --platform=linux/amd64 tiangolo/uvicorn-gunicorn:python3.11-slim

RUN apt-get update && apt-get install -y netcat-openbsd

WORKDIR /app/

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD exec uvicorn main:app --host 0.0.0.0 --port 80 --reload
