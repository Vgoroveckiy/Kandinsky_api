# client_con.py
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

logger = logging.getLogger(__name__)


class ConfigManager:
    """Управляет конфигурацией приложения, загружая настройки из переменных окружения."""

    def __init__(self):
        """
        Инициализирует конфигурацию, загружая переменные окружения.

        Загружает API ключи, параметры по умолчанию для генерации изображений.
        """
        logger.info("Initializing ConfigManager")
        self.api_key = os.getenv("FUSIONBRAIN_API_KEY")
        self.secret_key = os.getenv("FUSIONBRAIN_SECRET_KEY")
        self.prompt = os.getenv(
            "FUSIONBRAIN_DEFAULT_PROMPT", "Красивый закат на морском побережье"
        )
        self.width = int(os.getenv("FUSIONBRAIN_DEFAULT_WIDTH", 512))
        self.height = int(os.getenv("FUSIONBRAIN_DEFAULT_HEIGHT", 512))
        self.style = os.getenv("FUSIONBRAIN_DEFAULT_STYLE", None)
        self.negative_prompt = os.getenv("FUSIONBRAIN_DEFAULT_NEGATIVE_PROMPT", None)

    def validate(self) -> None:
        """
        Проверяет наличие обязательных ключей API.

        Raises:
            ValueError: Если отсутствует api_key или secret_key.
        """
        if not self.api_key or not self.secret_key:
            logger.error("API key or Secret key is missing in environment variables")
            raise ValueError("API key or Secret key is missing in environment variables.")
        logger.info("Configuration validated successfully")


class ImageHandler:
    """
    Обрабатывает сохранение изображений из данных API.

    Поддерживает загрузку изображений по URL или из base64-данных.
    """

    @staticmethod
    def save_image(image_data: str, save_path: str) -> None:
        """
        Сохраняет изображение на диск.

        Args:
            image_data (str): Данные изображения (URL или base64-строка).
            save_path (str): Путь для сохранения изображения.

        Raises:
            ValueError: Если формат данных изображения не поддерживается.
            requests.exceptions.RequestException: Если не удалось скачать изображение по URL.
            Exception: Для других ошибок, включая проблемы с декодированием base64.
        """
        try:
            logger.info("Saving image to %s", save_path)
            if isinstance(image_data, str):
                parsed_url = urlparse(image_data)
                if parsed_url.scheme in ("http", "https"):
                    response = requests.get(image_data)
                    response.raise_for_status()
                    with open(save_path, "wb") as file:
                        file.write(response.content)
                    logger.info("Image downloaded and saved to %s", save_path)
                else:
                    if image_data.startswith("data:image"):
                        image_data = image_data.split(",")[1]
                    image_bytes = base64.b64decode(image_data)
                    with open(save_path, "wb") as file:
                        file.write(image_bytes)
                    logger.info("Base64 image decoded and saved to %s", save_path)
            else:
                logger.error("Unsupported image data format: %s", type(image_data))
                raise ValueError("Unsupported image data format")
        except requests.exceptions.RequestException as e:
            logger.error("Failed to download image from URL %s: %s", image_data, e)
            raise
        except Exception as e:
            logger.error("Error saving image to %s: %s", save_path, e)
            raise


class FusionBrainAPI:
    """
    Клиент для взаимодействия с FusionBrain API для генерации изображений.

    Позволяет получать pipeline ID, проверять доступность сервиса, генерировать изображения и проверять статус генерации.
    """

    def __init__(self, url: str, api_key: str, secret_key: str):
        """
        Инициализирует клиент FusionBrain API.

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
        logger.info("FusionBrainAPI initialized with URL: %s", url)

    def get_pipeline(self) -> str:
        """
        Получает идентификатор pipeline из API.

        Returns:
            str: Идентификатор pipeline.

        Raises:
            requests.exceptions.RequestException: Если произошла сетевая ошибка.
            ValueError: Если ответ API имеет неожиданный формат.
            Exception: Для непредвиденных ошибок.
        """
        try:
            logger.info("Requesting pipeline ID from %skey/api/v1/pipelines", self.URL)
            response = requests.get(
                self.URL + "key/api/v1/pipelines", headers=self.AUTH_HEADERS
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or not data:
                raise ValueError(f"Unexpected pipeline response: {data}")
            pipeline_id = data[0]["id"]
            logger.info("Successfully retrieved pipeline ID: %s", pipeline_id)
            return pipeline_id
        except requests.exceptions.RequestException as e:
            logger.error("Network error in get_pipeline: %s", e)
            raise
        except ValueError as e:
            logger.error("Validation error in get_pipeline: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error in get_pipeline: %s", e)
            raise

    def check_availability(self, pipeline_id: str) -> dict:
        """
        Проверяет доступность сервиса.

        Args:
            pipeline_id (str): Идентификатор pipeline.

        Returns:
            dict: Информация о доступности сервиса.

        Raises:
            requests.exceptions.RequestException: Если произошла сетевая ошибка.
            Exception: Для непредвиденных ошибок.
        """
        try:
            logger.info("Checking service availability for pipeline %s", pipeline_id)
            response = requests.get(
                f"{self.URL}key/api/v1/pipeline/{pipeline_id}/availability",
                headers=self.AUTH_HEADERS,
            )
            response.raise_for_status()
            data = response.json()
            logger.info("Service availability status: %s", data)
            return data
        except requests.exceptions.RequestException as e:
            logger.error("Network error in check_availability: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error in check_availability: %s", e)
            raise

    def generate(
        self,
        prompt: str,
        pipeline: str,
        width: int = 1024,
        height: int = 1024,
        style: str = None,
        negative_prompt: str = None,
    ) -> str:
        """
        Инициирует генерацию изображения через API.

        Args:
            prompt (str): Текстовое описание для генерации изображения.
            pipeline (str): Идентификатор pipeline.
            width (int): Ширина изображения в пикселях.
            height (int): Высота изображения в пикселях.
            style (str, optional): Стиль изображения.
            negative_prompt (str, optional): Негативный промпт.

        Returns:
            str: UUID запроса генерации.

        Raises:
            requests.exceptions.RequestException: Если запрос завершился с ошибкой HTTP.
            ValueError: Если ответ API не содержит UUID.
            Exception: Для непредвиденных ошибок.
        """
        params = {
            "type": "GENERATE",
            "numImages": 1,
            "width": width,
            "height": height,
            "generateParams": {"query": f"{prompt}"},
        }

        if style:
            params["style"] = style

        if negative_prompt:
            params["negativePromptDecoder"] = negative_prompt

        data = {
            "pipeline_id": (None, pipeline),
            "params": (None, json.dumps(params), "application/json"),
        }

        try:
            logger.info(
                "Initiating image generation with prompt: %s, pipeline: %s, width: %d, height: %d, style: %s",
                prompt,
                pipeline,
                width,
                height,
                style if style else "default",
            )
            response = requests.post(
                self.URL + "key/api/v1/pipeline/run",
                headers=self.AUTH_HEADERS,
                files=data,
            )
            response.raise_for_status()

            data = response.json()
            if "uuid" not in data:
                logger.error("Unexpected generate response: %s", data)
                raise ValueError(f"Unexpected generate response: {data}")

            uuid = data["uuid"]
            logger.info("Image generation initiated, UUID: %s", uuid)
            return uuid
        except requests.exceptions.RequestException as e:
            logger.error("Network error in generate: %s", e)
            raise
        except ValueError as e:
            logger.error("Validation error in generate: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error in generate: %s", e)
            raise

    def check_generation(
        self,
        request_id: str,
        max_attempts: int = 10,
        initial_delay: float = 5,
        max_delay: float = 30,
    ) -> list:
        """
        Проверяет статус генерации изображения.

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
           logger.info("Checking generation status for UUID: %s", request_id)
           while attempt < max_attempts:
               response = requests.get(
                   self.URL + "key/api/v1/pipeline/status/" + request_id,
                   headers=self.AUTH_HEADERS,
               )
               response.raise_for_status()
               data = response.json()

               status = data.get("status")
               if status == "DONE":
                   files = data.get("result", {}).get("files", [])
                   censored = data.get("result", {}).get("censored", False)
                   if censored:
                       logging.warning(
                           "Content was censored for UUID: %s", request_id
                       )
                   if not files:
                       logging.warning(
                           "No files found in the generation result for UUID: %s",
                           request_id,
                       )
                   else:
                       logger.info(
                           "Generation completed, found %d files for UUID: %s",
                           len(files),
                           request_id,
                       )
                   return files
               elif status == "FAIL":
                   error_desc = data.get("errorDescription", "Unknown error")
                   logger.error("Generation failed: %s", error_desc)
                   raise Exception(f"Generation failed: {error_desc}")
               elif status in ["PROCESSING", "INITIAL"]:
                   logger.info("Generation status: %s, waiting...", status)
               else:
                   logging.warning("Unknown status: %s", status)

               attempt += 1
               logging.debug(
                   "Attempt %d/%d, retrying in %.2f seconds",
                   attempt,
                   max_attempts,
                   delay,
               )
               sleep(delay)
               delay = min(max_delay, delay * 2 + uniform(-0.5, 0.5))

           logger.error(
               "Generation did not complete in time for UUID: %s", request_id
           )
           raise TimeoutError("Generation did not complete in time.")
       except requests.exceptions.RequestException as e:
           logger.error(
               "Network error in check_generation for UUID %s: %s",
               request_id,
               e,
           )
           raise
       except Exception as e:
           logger.error(
               "Unexpected error in check_generation for UUID %s: %s",
               request_id,
               e,
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

       # Проверка доступности сервиса
       availability = api.check_availability(pipeline_id)
       if availability.get("pipeline_status") == "DISABLED_BY_QUEUE":
           logging.warning(
               "Service is currently unavailable due to high load. Try again later."
           )
           print(
               "Service is currently unavailable due to high load. Try again later."
           )