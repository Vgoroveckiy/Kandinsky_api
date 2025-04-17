import base64
import os
import time
import uuid
from io import BytesIO

import requests
from dotenv import load_dotenv
from PIL import Image

# Загружаем переменные окружения из файла .env
load_dotenv()


class FusionBrainClient:
    def __init__(self, api_key=None, secret_key=None):
        # Если ключи не переданы явно, берем их из переменных окружения
        self.api_key = api_key or os.getenv("FUSIONBRAIN_API_KEY")
        self.secret_key = secret_key or os.getenv("FUSIONBRAIN_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "API Key и Secret Key должны быть указаны либо в параметрах, либо в .env файле"
            )

        self.base_url = os.getenv(
            "FUSIONBRAIN_API_URL", "https://api-key.fusionbrain.ai/key/api/v1"
        )
        self.headers = {
            "X-Key": f"Key {self.api_key}",
            "X-Secret": f"Secret {self.secret_key}",
        }

    def get_models(self):
        """Получить список доступных моделей"""
        response = requests.get(f"{self.base_url}/models", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Ошибка при получении моделей: {response.status_code}, {response.text}"
            )

    def generate_image(
        self,
        prompt,
        model_id=None,
        width=None,
        height=None,
        images_num=None,
        style=None,
        negative_prompt="",
        guidance_scale=None,
        seed=None,
        save_path=None,
    ):
        """
        Генерация изображения по текстовому запросу

        :param prompt: Текстовый запрос для генерации изображения
        :param model_id: ID модели (берется из .env, если не указан)
        :param width: Ширина изображения (512, 768, 1024)
        :param height: Высота изображения (512, 768, 1024)
        :param images_num: Количество генерируемых изображений (1-4)
        :param style: Стиль изображения (опционально)
        :param negative_prompt: Негативный запрос (что не должно быть на изображении)
        :param guidance_scale: Степень соответствия запросу (1-12)
        :param seed: Seed для генерации (опционально)
        :param save_path: Путь для сохранения изображения (опционально)
        :return: Список сгенерированных изображений в формате PIL.Image
        """
        # Используем значения из .env, если параметры не переданы
        model_id = (
            model_id
            if model_id is not None
            else int(os.getenv("FUSIONBRAIN_MODEL_ID", 1))
        )
        width = (
            width if width is not None else int(os.getenv("FUSIONBRAIN_WIDTH", 1024))
        )
        height = (
            height if height is not None else int(os.getenv("FUSIONBRAIN_HEIGHT", 1024))
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
        save_path = (
            save_path
            if save_path is not None
            else os.getenv("FUSIONBRAIN_SAVE_PATH", "generated_images")
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
        payload = {
            "type": "TEXT2IMAGE",
            "num_images": images_num,
            "width": width,
            "height": height,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "model_id": model_id,
            "seed": seed,
        }

        if style:
            payload["style"] = style

        # Отправляем запрос на генерацию
        response = requests.post(
            f"{self.base_url}/text2image/run", headers=self.headers, json=payload
        )

        if response.status_code != 200:
            raise Exception(
                f"Ошибка при запросе генерации: {response.status_code}, {response.text}"
            )

        # Получаем UUID задачи
        task_uuid = response.json().get("uuid")

        # Проверяем статус задачи
        status = "PENDING"
        result = None

        while status == "PENDING":
            time.sleep(1)  # Ждем 1 секунду между запросами
            status_response = requests.get(
                f"{self.base_url}/text2image/status/{task_uuid}", headers=self.headers
            )

            if status_response.status_code != 200:
                raise Exception(
                    f"Ошибка при проверке статуса: {status_response.status_code}, {status_response.text}"
                )

            status_data = status_response.json()
            status = status_data.get("status")

            if status == "DONE":
                result = status_data.get("images")
            elif status == "FAILED":
                raise Exception(
                    f"Задача завершилась с ошибкой: {status_data.get('error')}"
                )

        # Обрабатываем результат
        images = []
        for i, img_base64 in enumerate(result):
            # Декодируем изображение из base64
            img_data = base64.b64decode(img_base64)
            img = Image.open(BytesIO(img_data))
            images.append(img)

            # Сохраняем изображение, если указан путь
            if save_path:
                # Создаем директорию, если она не существует
                os.makedirs(save_path, exist_ok=True)
                img_filename = f"{save_path}/image_{i+1}_{int(time.time())}.png"
                img.save(img_filename)
                print(f"Изображение сохранено: {img_filename}")

        return images


# Пример использования
if __name__ == "__main__":
    # Создаем клиент с ключами из .env файла
    client = FusionBrainClient()

    # Получение списка доступных моделей
    try:
        models = client.get_models()
        print("Доступные модели:")
        for model in models:
            print(f"ID: {model['id']}, Название: {model['name']}")
    except Exception as e:
        print(f"Ошибка при получении моделей: {e}")

    # Генерация изображения
    try:
        prompt = os.getenv(
            "FUSIONBRAIN_DEFAULT_PROMPT", "Красивый закат на морском побережье"
        )
        negative_prompt = os.getenv(
            "FUSIONBRAIN_NEGATIVE_PROMPT", "грязь, плохое качество, искажения"
        )

        images = client.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            # Остальные параметры будут взяты из .env
        )

        print(f"Сгенерировано изображений: {len(images)}")
    except Exception as e:
        print(f"Произошла ошибка при генерации: {e}")
