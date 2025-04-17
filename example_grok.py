import base64
import json
import logging
import os
from random import uniform
from time import sleep
from urllib.parse import urlparse

import requests
import requests.exceptions
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ConfigManager:
    """Управляет конфигурацией приложения, загружая настройки из переменных окружения."""

    def __init__(self):
        """Инициализирует конфигурацию, загружая переменные окружения."""
        logging.info("Initializing ConfigManager")
        self.api_key = os.getenv("FUSIONBRAIN_API_KEY")
        self.secret_key = os.getenv("FUSIONBRAIN_SECRET_KEY")
        self.prompt = os.getenv(
            "FUSIONBRAIN_DEFAULT_PROMPT", "Красивый закат на морском побережье"
        )
        self.width = int(os.getenv("FUSIONBRAIN_DEFAULT_WIDTH", 512))
        self.height = int(os.getenv("FUSIONBRAIN_DEFAULT_HEIGHT", 512))
        self.images = int(os.getenv("FUSIONBRAIN_DEFAULT_IMAGES", 1))

    def validate(self) -> None:
        """Проверяет наличие обязательных ключей API.

        Raises:
            ValueError: Если отсутствует api_key или secret_key.
        """
        if not self.api_key or not self.secret_key:
            logging.error("API key or Secret key is missing in environment variables")
            raise ValueError(
                "API key or Secret key is missing in environment variables."
            )
        logging.info("Configuration validated successfully")


class ImageHandler:
    """Обрабатывает сохранение изображений из данных API."""

    @staticmethod
    def save_image(image_data: str, save_path: str) -> None:
        """Сохраняет изображение на диск.

        Args:
            image_data (str): Данные изображения (URL или base64-строка).
            save_path (str): Путь для сохранения изображения.

        Raises:
            ValueError: Если формат данных изображения не поддерживается.
            requests.exceptions.RequestException: Если не удалось скачать изображение по URL.
            Exception: Для других ошибок, включая проблемы с декодированием base64.
        """
        try:
            logging.info("Saving image to %s", save_path)
            if isinstance(image_data, str):
                parsed_url = urlparse(image_data)
                if parsed_url.scheme in ("http", "https"):
                    response = requests.get(image_data)
                    response.raise_for_status()
                    with open(save_path, "wb") as file:
                        file.write(response.content)
                    logging.info("Image downloaded and saved to %s", save_path)
                else:
                    if image_data.startswith("data:image"):
                        image_data = image_data.split(",")[1]
                    image_bytes = base64.b64decode(image_data)
                    with open(save_path, "wb") as file:
                        file.write(image_bytes)
                    logging.info("Base64 image decoded and saved to %s", save_path)
            else:
                logging.error("Unsupported image data format: %s", type(image_data))
                raise ValueError("Unsupported image data format")
        except requests.exceptions.RequestException as e:
            logging.error("Failed to download image from URL %s: %s", image_data, e)
            raise
        except Exception as e:
            logging.error("Error saving image to %s: %s", save_path, e)
            raise


class FusionBrainAPI:
    """Клиент для взаимодействия с FusionBrain API для генерации изображений."""

    def __init__(self, url: str, api_key: str, secret_key: str):
        """Инициализирует клиент FusionBrain API.

        Args:
            url (str): Базовый URL API.
            api_key (str): Ключ API для аутентификации.
            secret_key (str): Секретный ключ API для аутентификации.
        """
        self.URL = url
        self.AUTH_HEADERS = {
            "X-Key": f"Key {api_key}",
            "X-Secret": f"Secret {secret_key}",
        }
        logging.info("FusionBrainAPI initialized with URL: %s", url)

    def get_pipeline(self) -> str:
        """Получает идентификатор pipeline из API.

        Returns:
            str: Идентификатор pipeline.

        Raises:
            requests.exceptions.RequestException: Если произошла сетевая ошибка.
            ValueError: Если ответ API имеет неожиданный формат.
            Exception: Для непредвиденных ошибок.
        """
        try:
            logging.info("Requesting pipeline ID from %skey/api/v1/pipelines", self.URL)
            response = requests.get(
                self.URL + "key/api/v1/pipelines", headers=self.AUTH_HEADERS
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or not data:
                raise ValueError(f"Unexpected pipeline response: {data}")
            pipeline_id = data[0]["id"]
            logging.info("Successfully retrieved pipeline ID: %s", pipeline_id)
            return pipeline_id
        except requests.exceptions.RequestException as e:
            logging.error("Network error in get_pipeline: %s", e)
            raise
        except ValueError as e:
            logging.error("Validation error in get_pipeline: %s", e)
            raise
        except Exception as e:
            logging.error("Unexpected error in get_pipeline: %s", e)
            raise

    def generate(
        self, prompt: str, pipeline: str, images: int, width: int, height: int
    ) -> str:
        """Инициирует генерацию изображения через API.

        Args:
            prompt (str): Текстовое описание для генерации изображения.
            pipeline (str): Идентификатор pipeline.
            images (int): Количество изображений для генерации.
            width (int): Ширина изображения в пикселях.
            height (int): Высота изображения в пикселях.

        Returns:
            str: UUID запроса генерации.

        Raises:
            Exception: Если запрос завершился с ошибкой HTTP.
            ValueError: Если ответ API не содержит UUID.
        """
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "generateParams": {"query": f"{prompt}"},
        }

        data = {
            "pipeline_id": (None, pipeline),
            "params": (None, json.dumps(params), "application/json"),
        }
        logging.info(
            "Initiating image generation with prompt: %s, pipeline: %s, images: %d, width: %d, height: %d",
            prompt,
            pipeline,
            images,
            width,
            height,
        )
        response = requests.post(
            self.URL + "key/api/v1/pipeline/run", headers=self.AUTH_HEADERS, files=data
        )

        if response.status_code not in [200, 201]:
            logging.error(
                "Failed to generate image. Status code: %d", response.status_code
            )
            raise Exception(
                f"Failed to generate image. Status code: {response.status_code}"
            )

        data = response.json()
        if "uuid" not in data:
            logging.error("Unexpected generate response: %s", data)
            raise ValueError(f"Unexpected generate response: {data}")

        uuid = data["uuid"]
        logging.info("Image generation initiated, UUID: %s", uuid)
        return uuid

    def check_generation(
        self,
        request_id: str,
        max_attempts: int = 10,
        initial_delay: float = 5,
        max_delay: float = 30,
    ) -> list:
        """Проверяет статус генерации изображения.

        Args:
            request_id (str): UUID запроса генерации.
            max_attempts (int): Максимальное количество попыток проверки статуса.
            initial_delay (float): Начальная задержка между попытками (в секундах).
            max_delay (float): Максимальная задержка между попытками (в секундах).

        Returns:
            list: Список данных сгенерированных изображений.

        Raises:
            requests.exceptions.RequestException: Если произошла сетевая ошибка.
            TimeoutError: Если генерация не завершилась в течение заданного времени.
            Exception: Для непредвиденных ошибок.
        """
        try:
            attempt = 0
            delay = initial_delay
            logging.info("Checking generation status for UUID: %s", request_id)
            while attempt < max_attempts:
                response = requests.get(
                    self.URL + "key/api/v1/pipeline/status/" + request_id,
                    headers=self.AUTH_HEADERS,
                )
                response.raise_for_status()
                data = response.json()

                if data["status"] == "DONE":
                    files = data.get("result", {}).get("files", [])
                    if not files:
                        logging.warning(
                            "No files found in the generation result for UUID: %s",
                            request_id,
                        )
                    else:
                        logging.info(
                            "Generation completed, found %d files for UUID: %s",
                            len(files),
                            request_id,
                        )
                    return files

                attempt += 1
                logging.debug(
                    "Attempt %d/%d failed, retrying in %.2f seconds",
                    attempt,
                    max_attempts,
                    delay,
                )
                sleep(delay)
                delay = min(max_delay, delay * 2 + uniform(-0.5, 0.5))

            logging.error(
                "Generation did not complete in time for UUID: %s", request_id
            )
            raise TimeoutError("Generation did not complete in time.")
        except requests.exceptions.RequestException as e:
            logging.error(
                "Network error in check_generation for UUID %s: %s", request_id, e
            )
            raise
        except Exception as e:
            logging.error(
                "Unexpected error in check_generation for UUID %s: %s", request_id, e
            )
            raise


if __name__ == "__main__":
    try:
        # Инициализация конфигурации
        config = ConfigManager()
        config.validate()

        # Инициализация API
        api = FusionBrainAPI(
            "https://api-key.fusionbrain.ai/", config.api_key, config.secret_key
        )

        # Получение pipeline ID
        pipeline_id = api.get_pipeline()

        # Генерация изображения
        uuid = api.generate(
            config.prompt, pipeline_id, config.images, config.width, config.height
        )

        # Проверка статуса генерации
        files = api.check_generation(uuid)

        # Проверка наличия файлов
        if not files:
            logging.warning("No image data found in API response")
            print("No image data found. Check the API response for errors.")
        else:
            # Создаем папку output, если она не существует
            os.makedirs("output", exist_ok=True)
            logging.info("Created output directory")

            # Сохранение изображений
            image_handler = ImageHandler()
            for i, file_data in enumerate(files):
                save_path = os.path.join("output", f"generated_image_{i + 1}.png")
                image_handler.save_image(file_data, save_path)
    except Exception as e:
        logging.error("An error occurred: %s", e)
        print(f"An error occurred: {e}")
