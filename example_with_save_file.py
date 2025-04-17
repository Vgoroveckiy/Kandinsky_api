import base64
import json
import os
import time
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()


class FusionBrainAPI:
    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            "X-Key": f"Key {api_key}",
            "X-Secret": f"Secret {secret_key}",
        }

    def get_pipeline(self):
        try:
            response = requests.get(
                self.URL + "key/api/v1/pipelines", headers=self.AUTH_HEADERS
            )
            print("DEBUG: status", response.status_code)

            data = response.json()
            if not isinstance(data, list) or not data:
                raise ValueError(f"Unexpected pipeline response: {data}")

            return data[0]["id"]
        except Exception as e:
            print(f"Error in get_pipeline: {e}")
            raise

    def generate(self, prompt, pipeline, images, width, height):
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
        response = requests.post(
            self.URL + "key/api/v1/pipeline/run", headers=self.AUTH_HEADERS, files=data
        )

        # Проверяем успешность запроса
        if response.status_code not in [200, 201]:
            raise Exception(
                f"Failed to generate image. Status code: {response.status_code}"
            )

        data = response.json()
        if "uuid" not in data:
            raise ValueError(f"Unexpected generate response: {data}")

        return data["uuid"]

    def check_generation(self, request_id, attempts=10, delay=10):
        try:
            while attempts > 0:
                response = requests.get(
                    self.URL + "key/api/v1/pipeline/status/" + request_id,
                    headers=self.AUTH_HEADERS,
                )

                # Проверяем успешность запроса
                if response.status_code != 200:
                    raise Exception(
                        f"Failed to check generation status. Status code: {response.status_code}"
                    )

                data = response.json()

                if data["status"] == "DONE":
                    files = data.get("result", {}).get("files", [])
                    if not files:
                        print("WARNING: No files found in the generation result.")
                    return files

                attempts -= 1
                time.sleep(delay)

            raise TimeoutError("Generation did not complete in time.")
        except Exception as e:
            print(f"Error in check_generation: {e}")
            raise

    def save_image(self, image_data, save_path):
        """Сохраняет изображение по URL или base64-строке."""
        try:
            # Проверяем, является ли строка валидным URL
            parsed_url = urlparse(image_data)
            if parsed_url.scheme and parsed_url.netloc:
                # Если это URL, скачиваем изображение
                response = requests.get(image_data)
                if response.status_code == 200:
                    with open(save_path, "wb") as file:
                        file.write(response.content)
                    print(f"Image saved to {save_path}")
                else:
                    raise Exception(f"Failed to download image from {image_data}")
            else:
                # Если это base64-строка, декодируем и сохраняем
                try:
                    # Проверяем, есть ли префикс "data:image". Если нет, используем данные как есть.
                    if image_data.startswith("data:image"):
                        image_data = image_data.split(",")[1]  # Убираем префикс
                    # Декодируем base64
                    image_bytes = base64.b64decode(image_data)
                    with open(save_path, "wb") as file:
                        file.write(image_bytes)
                    print(f"Image saved to {save_path}")
                except Exception as decode_error:
                    raise Exception(
                        f"Failed to decode and save base64 image: {decode_error}"
                    )
        except Exception as e:
            print(f"Error in save_image: {e}")
            raise


if __name__ == "__main__":
    try:
        # Загрузка ключей из переменных окружения
        api_key = os.getenv("FUSIONBRAIN_API_KEY")
        secret_key = os.getenv("FUSIONBRAIN_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "API key or Secret key is missing in environment variables."
            )

        # Инициализация API
        api = FusionBrainAPI("https://api-key.fusionbrain.ai/", api_key, secret_key)

        # Загрузка настроек из переменных окружения
        prompt = os.getenv(
            "FUSIONBRAIN_DEFAULT_PROMPT", "Красивый закат на морском побережье"
        )

        width = int(os.getenv("FUSIONBRAIN_DEFAULT_WIDTH", 512))
        height = int(os.getenv("FUSIONBRAIN_DEFAULT_HEIGHT", 512))
        images = int(os.getenv("FUSIONBRAIN_DEFAULT_IMAGES", 1))

        # Получение pipeline ID
        pipeline_id = api.get_pipeline()

        # Генерация изображения
        uuid = api.generate(prompt, pipeline_id, images, width, height)

        # Проверка статуса генерации
        files = api.check_generation(uuid)

        # Проверка наличия файлов
        if not files:
            print("No image data found. Check the API response for errors.")
        else:
            # Создаем папку output, если она не существует
            os.makedirs("output", exist_ok=True)

            # Сохранение изображений
            for i, file_data in enumerate(files):
                save_path = os.path.join("output", f"generated_image_{i + 1}.png")
                api.save_image(file_data, save_path)
    except Exception as e:
        print(f"An error occurred: {e}")
