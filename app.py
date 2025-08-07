# app.py
import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from threading import Thread

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory, session
from werkzeug.utils import secure_filename

# Импортируем классы из существующего client_con.py
from client_con import ConfigManager, FusionBrainAPI, ImageHandler
from flask_session import Session

# Загружаем переменные из .env
load_dotenv()

# Получаем параметры логирования из .env
log_file = os.getenv("LOG_FILE", "app.log")  # По умолчанию app.log
log_max_size_mb = float(os.getenv("LOG_MAX_SIZE_MB", 1))  # По умолчанию 1 МБ
log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 3))  # По умолчанию 3 файлов

# Преобразуем размер в байты (1 МБ = 1024 * 1024 байт)
log_max_size = int(log_max_size_mb * 1024 * 1024)

# Получаем параметр очистки output из .env
output_cleanup_age_hours = float(
    os.getenv("OUTPUT_CLEANUP_AGE_HOURS", 24)
)  # По умолчанию 24 часа

# Настройка корневого логгера
if not logging.getLogger("").handlers:
    logging.getLogger("").setLevel(logging.INFO)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=log_max_size, backupCount=log_backup_count, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger("").addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger("").addHandler(console_handler)

# Отключение логов Werkzeug для HTTP-запросов
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.disabled = True  # Полностью отключаем логгер werkzeug
werkzeug_logger.handlers = []  # Удаляем любые обработчики
werkzeug_logger.propagate = False  # Не передаём сообщения корневому логгеру

# Логгер для текущего модуля
logger = logging.getLogger(__name__)

# Тестовое сообщение для проверки
logger.info("Logging initialized")


def cleanup_output_folder():
    """
    Очищает папку output от старых подкаталогов.

    Подкаталоги удаляются, если они старше OUTPUT_CLEANUP_AGE_HOURS.
    Логгирует информацию о процессе очистки.
    """
    output_folder = os.path.abspath(app.config["UPLOAD_FOLDER"])
    cleanup_age_seconds = output_cleanup_age_hours * 3600  # Конвертируем часы в секунды
    current_time = time.time()

    logger.info(
        f"Starting cleanup of {output_folder} (age > {output_cleanup_age_hours} hours)"
    )

    try:
        # Проверяем, существует ли каталог
        if not os.path.exists(output_folder):
            logger.info(
                f"Output folder {output_folder} does not exist, skipping cleanup"
            )
            return

        # Перебираем подкаталоги
        deleted_count = 0
        for subdir in os.listdir(output_folder):
            subdir_path = os.path.join(output_folder, subdir)
            # Проверяем, что это каталог
            if not os.path.isdir(subdir_path):
                continue

            # Получаем время последней модификации каталога
            mtime = os.path.getmtime(subdir_path)
            age_seconds = current_time - mtime

            # Удаляем, если каталог старше заданного возраста
            if age_seconds > cleanup_age_seconds:
                try:
                    shutil.rmtree(
                        subdir_path
                    )  # Рекурсивно удаляем каталог и его содержимое
                    logger.info(
                        f"Deleted old directory: {subdir_path} (age: {age_seconds/3600:.1f} hours)"
                    )
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete directory {subdir_path}: {e}")

        logger.info(f"Cleanup completed: deleted {deleted_count} directories")

    except Exception as e:
        logger.error(f"Error during output folder cleanup: {e}")


def schedule_cleanup():
    """
    Запускает функцию cleanup_output_folder каждые 10 минут в фоновом потоке.

    Работает бесконечно в цикле.
    """
    while True:
        cleanup_output_folder()
        time.sleep(600)  # Ждём 10 минут (600 секунд)


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fusionbrain-flask-app-secret")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["UPLOAD_FOLDER"] = "output"
Session(app)

# Словарь для хранения статусов задач
tasks = {}

# Запуск фоновой задачи очистки при старте приложения
cleanup_thread = Thread(target=schedule_cleanup, daemon=True)
cleanup_thread.start()
logger.info("Started output folder cleanup thread")


def generate_image_task(task_id, prompt, width, height, style, negative_prompt):
    """
    Фоновая задача для генерации изображения по заданному промпту.

    Args:
        task_id (str): Уникальный идентификатор задачи.
        prompt (str): Текстовый промпт для генерации изображения.
        width (int): Ширина изображения.
        height (int): Высота изображения.
        style (str): Стиль генерации изображения.
        negative_prompt (str): Отрицательный промпт для ограничений.

    Возвращает:
        None. Результат сохраняется в tasks.
    """
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
            image_path = f"{task_id}/{filename}"
            image_url = f"/image/{task_id}/{filename}"
            logger.info(f"Image saved: path={image_path}, url={image_url}")
            image_paths.append(
                {
                    "path": image_path,
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
        logger.error(f"Error in task {task_id}: {e}")


@app.route("/")
def index():
    """Главная страница приложения"""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    Запускает задачу генерации изображения.

    Args:
        request: POST-запрос с данными формы.

    Возвращает:
        JSON: Статус задачи и task_id.
    """
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
        logger.error(f"Error starting generation task: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/task/<task_id>", methods=["GET"])
def task_status(task_id):
    """
    Возвращает статус текущей задачи.

    Args:
        task_id (str): Идентификатор задачи.

    Возвращает:
        JSON: Статус задачи и связанные данные.
    """
    if task_id not in tasks:
        return jsonify({"success": False, "error": "Task not found"}), 404

    task_data = tasks[task_id]
    if "image_paths" in task_data:
        for img in task_data["image_paths"]:
            original_path = img["path"]
            img["path"] = img["path"].replace("\\", "/")
            logger.info(
                f"Task {task_id} image path: original={original_path}, normalized={img['path']}"
            )
    # Логируем только завершение или ошибки
    if task_data.get("status") in ["completed", "error"]:
        logger.info(
            f"Task {task_id} status: {task_data['status']}, progress: {task_data['progress']}"
        )
    return jsonify({"success": True, "task": task_data})


@app.route("/image/<task_id>/<filename>")
def serve_image(task_id, filename):
    """
    Отправляет сгенерированное изображение по указанному пути.

    Args:
        task_id (str): Идентификатор задачи.
        filename (str): Имя файла изображения.

    Возвращает:
         Отправляет файл изображения.
     """
     task_folder = os.path.join(app.config["UPLOAD_FOLDER"], task_id)
     return send_from_directory(task_folder, filename)


@app.route("/download/<task_id>/<filename>")
def download_image(task_id, filename):
    """
     Обрабатывает запрос на скачивание сгенерированного изображения.

     Args:
         task_id (str): Идентификатор задачи.
         filename (str): Имя файла изображения.

     Возвращает:
         Отправляет файл как аттачмент.
     """
     task_folder = os.path.join(app.config["UPLOAD_FOLDER"], task_id)
     return send_from_directory(
         task_folder,
         filename,
         as_attachment=True,
         download_name=secure_filename(filename),
     )


@app.route("/styles")
def get_styles():
     """
     Возвращает список доступных стилей генерации изображений.

     Возвращает:
         JSON: Список стилей с идентификаторами и названиями.
     """
     styles = [
         {"id": "DEFAULT", "name": "По умолчанию"},
         {"id": "ANIME", "name": "Аниме"},
         {"id": "PORTRAIT", "name": "Портрет"},
         {"id": "REALISTIC", "name": "Реалистичный"},
         {"id": "UHD", "name": "Ультра HD"},
     ]
     return jsonify(styles)


if __name__ == "__main__":
     """
     Основной блок запуска приложения.

     Выполняет проверку конфигурации перед запуском сервера.
     """
     try:
         config = ConfigManager()
         config.validate()
         os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
         app.run(host="0.0.0.0", port=5000, debug=True)
     except ValueError as e:
         logger.error(f"Configuration error: {e}")
         print(f"Error: {e}")
         print("Please check your API credentials in .env file")