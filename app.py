# app.py
import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime
from threading import Thread

from flask import Flask, jsonify, render_template, request, send_from_directory, session
from flask_session import Session
from werkzeug.utils import secure_filename

# Импортируем классы из существующего client_con.py
from client_con import ConfigManager, FusionBrainAPI, ImageHandler

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fusionbrain-flask-app-secret")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["UPLOAD_FOLDER"] = "output"
Session(app)

# Словарь для хранения статусов задач
tasks = {}


def generate_image_task(task_id, prompt, width, height, style, negative_prompt):
    """Фоновая задача для генерации изображения"""
    try:
        # Обновляем статус задачи
        tasks[task_id] = {"status": "initializing", "progress": 10}

        # Инициализация конфигурации
        config = ConfigManager()

        # Переопределяем параметры из запроса
        config.prompt = prompt
        config.width = width
        config.height = height
        config.style = style
        config.negative_prompt = negative_prompt

        # Проверяем конфигурацию
        config.validate()

        # Обновляем статус
        tasks[task_id]["status"] = "connecting"
        tasks[task_id]["progress"] = 20

        # Инициализация API
        api = FusionBrainAPI(
            "https://api-key.fusionbrain.ai/", config.api_key, config.secret_key
        )

        # Получение pipeline ID
        tasks[task_id]["status"] = "getting_pipeline"
        tasks[task_id]["progress"] = 30
        pipeline_id = api.get_pipeline()

        # Проверка доступности сервиса
        tasks[task_id]["status"] = "checking_availability"
        tasks[task_id]["progress"] = 40
        availability = api.check_availability(pipeline_id)
        if availability.get("pipeline_status") == "DISABLED_BY_QUEUE":
            tasks[task_id]["status"] = "unavailable"
            tasks[task_id][
                "message"
            ] = "Сервис временно недоступен из-за высокой нагрузки"
            return

        # Генерация изображения
        tasks[task_id]["status"] = "generating"
        tasks[task_id]["progress"] = 50
        generation_uuid = api.generate(
            config.prompt,
            pipeline_id,
            config.width,
            config.height,
            style=config.style,
            negative_prompt=config.negative_prompt,
        )

        # Проверка статуса генерации
        tasks[task_id]["status"] = "checking_generation"
        tasks[task_id]["progress"] = 70
        files = api.check_generation(generation_uuid)

        # Проверка наличия файлов
        if not files:
            tasks[task_id]["status"] = "no_files"
            tasks[task_id][
                "message"
            ] = "Изображения не получены. Проверьте журнал ошибок."
            return

        # Сохранение изображений
        tasks[task_id]["status"] = "saving"
        tasks[task_id]["progress"] = 90

        # Создаем папку output, если она не существует
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

        # Создаем подпапку для задачи
        task_folder = os.path.join(app.config["UPLOAD_FOLDER"], task_id)
        os.makedirs(task_folder, exist_ok=True)

        image_handler = ImageHandler()
        image_paths = []

        for i, file_data in enumerate(files):
            filename = f"generated_{int(time.time())}_{i + 1}.png"
            save_path = os.path.join(task_folder, filename)
            image_handler.save_image(file_data, save_path)
            image_paths.append(
                {
                    "path": os.path.join(task_id, filename),
                    "url": f"/image/{task_id}/{filename}",
                }
            )

        # Задача завершена успешно
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["image_paths"] = image_paths

    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["message"] = str(e)
        logging.error(f"Error in task {task_id}: {e}")


@app.route("/")
def index():
    """Главная страница приложения"""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Маршрут для запуска генерации изображения"""
    try:
        # Получаем параметры из формы
        prompt = request.form["prompt"]
        width = int(request.form.get("width", 512))
        height = int(request.form.get("height", 512))
        style = request.form.get("style") or None
        negative_prompt = request.form.get("negative_prompt") or None

        # Создаем уникальный идентификатор задачи
        task_id = str(uuid.uuid4())

        # Инициализируем информацию о задаче
        tasks[task_id] = {
            "status": "created",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "params": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "style": style,
                "negative_prompt": negative_prompt,
            },
        }

        # Запускаем задачу в отдельном потоке
        thread = Thread(
            target=generate_image_task,
            args=(task_id, prompt, width, height, style, negative_prompt),
        )
        thread.daemon = True
        thread.start()

        return jsonify({"success": True, "task_id": task_id})

    except Exception as e:
        logging.error(f"Error starting generation task: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/task/<task_id>", methods=["GET"])
def task_status(task_id):
    """API-маршрут для получения статуса задачи"""
    if task_id not in tasks:
        return jsonify({"success": False, "error": "Task not found"}), 404

    return jsonify({"success": True, "task": tasks[task_id]})


@app.route("/image/<task_id>/<filename>")
def serve_image(task_id, filename):
    """Маршрут для отдачи сгенерированного изображения"""
    task_folder = os.path.join(app.config["UPLOAD_FOLDER"], task_id)
    return send_from_directory(task_folder, filename)


@app.route("/download/<task_id>/<filename>")
def download_image(task_id, filename):
    """Маршрут для скачивания изображения"""
    task_folder = os.path.join(app.config["UPLOAD_FOLDER"], task_id)
    return send_from_directory(
        task_folder,
        filename,
        as_attachment=True,
        download_name=secure_filename(filename),
    )


@app.route("/styles")
def get_styles():
    """Маршрут для получения списка доступных стилей (заглушка)"""
    styles = [
        {"id": "DEFAULT", "name": "По умолчанию"},
        {"id": "ANIME", "name": "Аниме"},
        {"id": "PORTRAIT", "name": "Портрет"},
        {"id": "REALISTIC", "name": "Реалистичный"},
        {"id": "UHD", "name": "Ультра HD"},
    ]
    return jsonify(styles)


if __name__ == "__main__":
    # Проверка наличия конфигурации перед запуском
    try:
        config = ConfigManager()
        config.validate()
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        app.run(host="0.0.0.0", port=5000, debug=True)
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        print(f"Error: {e}")
        print("Please check your API credentials in .env file")
