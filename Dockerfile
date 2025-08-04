# Сборка и запуск приложения
FROM python:3.11-slim
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы приложения
COPY . .

# Создаём необходимые каталоги
RUN mkdir -p /app/output && \
    mkdir -p /app/flask_session

# CMD для запуска приложения
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]