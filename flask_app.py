import base64
import json
import logging
import os
import threading
import time
import uuid
from io import BytesIO

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


# Модифицированный клиент FusionBrain с поддержкой отслеживания прогресса
"""
Клиент для взаимодействия с API FusionBrain.

Этот класс предоставляет функциональность для взаимодействия с API FusionBrain для генерации изображений из текстовых описаний.
Он обрабатывает аутентификацию, выбор модели, генерацию изображений и отслеживание прогресса.

Атрибуты:
    api_key (str): Ключ API для сервиса FusionBrain.
    secret_key (str): Секретный ключ для сервиса FusionBrain.
    base_url (str): Базовый URL для API FusionBrain.
    headers (dict): HTTP-заголовки для запросов API.
    tasks_progress (dict): Словарь для отслеживания прогресса задач генерации.
    tasks_results (dict): Словарь для хранения результатов выполненных задач.

Raises:
    ValueError: Если ключ API или секретный ключ не были указаны ни в параметрах, ни в переменных окружения.
"""


class FusionBrainClient:
    def __init__(self, api_key=None, secret_key=None):
        # Загрузка переменных окружения
        """
        Инициализирует клиент FusionBrain.

        Args:
            api_key (str, optional): API Key для FusionBrain. Defaults to None.
            secret_key (str, optional): Secret Key для FusionBrain. Defaults to None.

        Raises:
            ValueError: Если API Key и Secret Key не были указаны ни в параметрах, ни в переменных окружения.
        """
        load_dotenv()

        # Если ключи не переданы явно, берём их из переменных окружения
        self.api_key = api_key or os.getenv("FUSIONBRAIN_API_KEY")
        self.secret_key = secret_key or os.getenv("FUSIONBRAIN_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "API Key и Secret Key должны быть указаны либо в параметрах, либо в .env файле"
            )

        # Исправленный базовый URL
        self.base_url = os.getenv(
            "FUSIONBRAIN_API_URL", "https://api-key.fusionbrain.ai/"
        )
        self.headers = {
            "X-Key": f"Key {self.api_key}",
            "X-Secret": f"Secret {self.secret_key}",
        }
        # Словарь для хранения прогресса выполнения задач
        self.tasks_progress = {}
        # Словарь для хранения результатов генерации
        self.tasks_results = {}

    def get_models(self):
        """
        Получает список моделей, доступных в FusionBrain.

        Returns:
            list: Список моделей в формате [{"id": <id>, "name": <name>}, ...].

        Raises:
            ValueError: Если не удалось получить список моделей.
        """
        try:
            logging.info(
                "Requesting pipelines from %skey/api/v1/pipelines", self.base_url
            )
            response = requests.get(
                f"{self.base_url}key/api/v1/pipelines", headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            print("Pipelines response:", data)  # Для отладки
            if not isinstance(data, list):
                logging.error("Unexpected pipelines response format: %s", data)
                raise ValueError(f"Unexpected pipelines response: {data}")

            # Формируем список моделей в формате, подходящем для index.html
            models = [
                {
                    "id": pipeline.get("id", i + 1),
                    "name": pipeline.get("name", f"Model {i + 1}"),
                }
                for i, pipeline in enumerate(data)
            ]
            logging.info("Retrieved %d pipelines: %s", len(models), models)
            return models
        except requests.exceptions.RequestException as e:
            logging.error("Failed to fetch pipelines: %s", e)
            # Возвращаем заглушку при ошибке
            return [{"id": "kandinsky_3.1", "name": "Kandinsky 3.1"}]
        except Exception as e:
            logging.error("Unexpected error in get_models: %s", e)
            return [{"id": "kandinsky_3.1", "name": "Kandinsky 3.1"}]

    def get_task_progress(self, task_id):
        """Получить текущий прогресс выполнения задачи"""
        return self.tasks_progress.get(task_id, {"status": "UNKNOWN", "progress": 0})

    def get_task_result(self, task_id):
        """Получить результат выполнения задачи"""
        return self.tasks_results.get(task_id)

    def generate_image_async(
        self,
        prompt,
        task_id=None,
        model_id=None,
        width=None,
        height=None,
        images_num=None,
        style=None,
        negative_prompt="",
        guidance_scale=None,
        seed=None,
    ):
        """
        Асинхронная генерация изображения по текстовому запросу

        :param task_id: Уникальный идентификатор задачи для отслеживания прогресса
        :return: task_id для отслеживания прогресса генерации
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        # Устанавливаем начальный прогресс
        self.tasks_progress[task_id] = {"status": "PENDING", "progress": 0}

        # Запускаем генерацию в отдельном потоке
        thread = threading.Thread(
            target=self._generate_image_thread,
            args=(
                prompt,
                task_id,
                model_id,
                width,
                height,
                images_num,
                style,
                negative_prompt,
                guidance_scale,
                seed,
            ),
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _generate_image_thread(
        self,
        prompt,
        task_id,
        model_id=None,
        width=None,
        height=None,
        images_num=None,
        style=None,
        negative_prompt="",
        guidance_scale=None,
        seed=None,
    ):
        """Внутренний метод для генерации изображения в отдельном потоке"""
        try:
            # Используем значения из .env, если параметры не переданы
            model_id = (
                model_id
                if model_id is not None
                else os.getenv("FUSIONBRAIN_MODEL_ID", "kandinsky_3.1")
            )
            width = (
                width
                if width is not None
                else int(os.getenv("FUSIONBRAIN_WIDTH", 1024))
            )
            height = (
                height
                if height is not None
                else int(os.getenv("FUSIONBRAIN_HEIGHT", 1024))
            )
            images_num = (
                images_num
                if images_num is not None
                else int(os.getenv("FUSIONBRAIN_IMAGES_NUM", 1))
            )
            guidance_scale = (
                guidance_scale
                if guidance_scale is not None
                else float(os.getenv("FUSIONBRAIN_GUIDANCE_SCALE", 7))
            )

            # Проверка параметров
            if width not in [512, 768, 1024]:
                raise ValueError("Width must be one of: 512, 768, 1024")
            if height not in [512, 768, 1024]:
                raise ValueError("Height must be one of: 512, 768, 1024")
            if images_num < 1 or images_num > 4:
                raise ValueError("images_num must be between 1 and 4")
            if guidance_scale < 1 or guidance_scale > 12:
                raise ValueError("guidance_scale must be between 1 and 12")

            # Если seed не указан, генерируем случайный
            if seed is None:
                seed = uuid.uuid4().int % (2**32)

            # Формируем данные запроса
            params = {
                "type": "GENERATE",
                "numImages": images_num,
                "width": width,
                "height": height,
                "negativePromptDecoder": negative_prompt,
                "generateParams": {
                    "query": prompt,
                },
            }
            if style:
                params["generateParams"]["style"] = style

            data = {
                "pipeline_id": (None, model_id),
                "params": (None, json.dumps(params), "application/json"),
            }

            # Обновляем прогресс - задача отправлена
            self.tasks_progress[task_id] = {"status": "SENDING", "progress": 10}

            # Отправляем запрос на генерацию
            logging.info("Sending generate request with pipeline_id: %s", model_id)
            response = requests.post(
                f"{self.base_url}key/api/v1/pipeline/run",
                headers=self.headers,
                files=data,
            )
            print("Generate response:", response.text)  # Для отладки

            if response.status_code != 200:
                error_msg = response.json().get("error", response.text)
                self.tasks_progress[task_id] = {
                    "status": "FAILED",
                    "progress": 0,
                    "error": f"Ошибка при запросе генерации: {response.status_code}, {error_msg}",
                }
                return

            # Получаем UUID задачи от API
            api_task_uuid = response.json().get("uuid")
            if not api_task_uuid:
                self.tasks_progress[task_id] = {
                    "status": "FAILED",
                    "progress": 0,
                    "error": "API response does not contain uuid",
                }
                return

            # Обновляем прогресс - запрос принят
            self.tasks_progress[task_id] = {"status": "PROCESSING", "progress": 30}

            # Проверяем статус задачи
            status = "PENDING"
            result = None
            start_time = time.time()
            check_count = 0

            while status == "PENDING":
                time.sleep(1)  # Ждем 1 секунду между запросами
                check_count += 1

                status_response = requests.get(
                    f"{self.base_url}key/api/v1/pipeline/status/{api_task_uuid}",
                    headers=self.headers,
                )
                print("Status response:", status_response.text)  # Для отладки

                if status_response.status_code != 200:
                    error_msg = status_response.json().get(
                        "error", status_response.text
                    )
                    self.tasks_progress[task_id] = {
                        "status": "FAILED",
                        "progress": 0,
                        "error": f"Ошибка при проверке статуса: {status_response.status_code}, {error_msg}",
                    }
                    return

                status_data = status_response.json()
                status = status_data.get("status")

                # Прогресс от 30% до 90% в зависимости от времени ожидания
                elapsed = time.time() - start_time
                progress_percent = min(90, 30 + (check_count * 5))

                self.tasks_progress[task_id] = {
                    "status": "PROCESSING",
                    "progress": progress_percent,
                }

                if status == "DONE":
                    result = status_data.get("result", {}).get("files", [])
                    break
                elif status == "FAILED":
                    self.tasks_progress[task_id] = {
                        "status": "FAILED",
                        "progress": 0,
                        "error": f"Задача завершилась с ошибкой: {status_data.get('error')}",
                    }
                    return

            # Обрабатываем результат
            if result:
                images = []
                for i, file_url in enumerate(result):
                    # Загружаем изображение по URL
                    img_response = requests.get(file_url)
                    img_response.raise_for_status()
                    img = Image.open(BytesIO(img_response.content))
                    img_base64 = base64.b64encode(img_response.content).decode("utf-8")
                    images.append({"index": i, "base64": img_base64})

                # Сохраняем результат
                self.tasks_results[task_id] = images

                # Обновляем статус - задача выполнена успешно
                self.tasks_progress[task_id] = {"status": "COMPLETED", "progress": 100}
            else:
                self.tasks_progress[task_id] = {
                    "status": "FAILED",
                    "progress": 0,
                    "error": "Не удалось получить результат генерации",
                }

        except Exception as e:
            self.tasks_progress[task_id] = {
                "status": "FAILED",
                "progress": 0,
                "error": str(e),
            }


# Инициализация Flask приложения
app = Flask(__name__)
app.config["STATIC_FOLDER"] = "static"
app.config["UPLOAD_FOLDER"] = "generated_images"

# Создаём директорию для сохранения изображений, если она не существует
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Создаём клиент FusionBrain
client = FusionBrainClient()


@app.route("/")
def index():
    """
    Главная страница.
    Отображает форму для ввода текстового описания и списка доступных моделей.
    """
    try:
        # Получаем список моделей для отображения в форме
        models = client.get_models()
    except Exception as e:
        models = [{"id": "kandinsky_3.1", "name": "Kandinsky 3.1"}]
        logging.error("Ошибка при получении моделей: %s", e)

    return render_template("index.html", models=models)


@app.route("/generate", methods=["POST"])
def generate():
    # Получаем параметры из формы
    """
    Обрабатывает POST-запросы для генерации изображений.

    Эндпоинт получает параметры генерации (промпт, отрицательный промпт, ID модели, размеры и т.д.)
    из отправленной формы и запускает асинхронную задачу генерации изображений. Возвращает ID задачи
    для отслеживания прогресса генерации.

    Аргументы (данные формы):
        prompt (str): Обязательно. Текстовый промпт для генерации изображения.
        negative_prompt (str): Необязательно. Отрицательный промпт для исключения из генерации.
        model_id (str): Обязательно. ID модели, используемой для генерации.
        width (int): Необязательно. Ширина сгенерированного изображения (по умолчанию: 1024).
        height (int): Необязательно. Высота сгенерированного изображения (по умолчанию: 1024).
        images_num (int): Необязательно. Количество изображений для генерации (по умолчанию: 1).
        style (str): Необязательно. Стиль, применяемый к сгенерированным изображениям.

    Возвращает:
        JSON: Содержит task_id для асинхронной задачи генерации в случае успеха,
              или сообщение об ошибке, если проверка не удалась.

    Исключения:
        400 Bad Request: Если отсутствуют обязательные параметры (prompt или model_id).
    """
    prompt = request.form.get("prompt", "")
    negative_prompt = request.form.get("negative_prompt", "")
    model_id = request.form.get("model_id", "")
    width = int(request.form.get("width", 1024))
    height = int(request.form.get("height", 1024))
    images_num = int(request.form.get("images_num", 1))
    style = request.form.get("style", "")

    logging.info("Generate request: model_id=%s, prompt=%s", model_id, prompt)

    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Запускаем асинхронную генерацию
    task_id = client.generate_image_async(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=model_id,
        width=width,
        height=height,
        images_num=images_num,
        style=style if style else None,
    )

    return jsonify({"task_id": task_id})


@app.route("/progress/<task_id>")
def progress(task_id):
    # Получаем прогресс выполнения задачи
    progress_data = client.get_task_progress(task_id)
    return jsonify(progress_data)


@app.route("/result/<task_id>")
def result(task_id):
    # Получаем результат выполнения задачи
    result_data = client.get_task_result(task_id)
    if result_data:
        return jsonify({"status": "success", "images": result_data})
    else:
        return jsonify({"status": "not_found"})


@app.route("/save/<task_id>/<int:image_index>", methods=["POST"])
def save_image(task_id, image_index):
    """
    Сохраняет выбранное изображение на диск.

    Args:
        task_id (str): Идентификатор задачи.
        image_index (int): Индекс изображения в списке результатов.

    Returns:
        JSON: Имя файла, если изображение сохранено, или сообщение об ошибке.
    """
    result_data = client.get_task_result(task_id)

    if not result_data or image_index >= len(result_data):
        return jsonify({"status": "error", "message": "Изображение не найдено"})

    try:
        # Получаем base64 строку выбранного изображения
        img_base64 = result_data[image_index]["base64"]

        # Декодируем изображение из base64
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data))

        # Создаём имя файла с временной меткой
        filename = f"fusionbrain_{int(time.time())}_{image_index}.png"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Сохраняем изображение
        img.save(filepath)

        return jsonify(
            {
                "status": "success",
                "message": "Изображение сохранено",
                "filename": filename,
            }
        )
    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"Ошибка при сохранении: {str(e)}"}
        )


@app.route("/downloads/<path:filename>")
def download_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"], filename, as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)
