# Этап сборки
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Финальный образ
FROM python:3.11-slim
WORKDIR /app

# Создаём необходимые каталоги
RUN mkdir -p /app/output && \
    mkdir -p /app/flask_session

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
