import json
import os
import time

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
        response = requests.get(
            self.URL + "key/api/v1/pipelines", headers=self.AUTH_HEADERS
        )
        print("DEBUG: status", response.status_code)
        print("DEBUG: response text", response.text)

        data = response.json()
        if not isinstance(data, list) or not data:
            raise ValueError(f"Unexpected pipeline response: {data}")

        return data[0]["id"]

    def generate(self, prompt, pipeline, images=1, width=512, height=512):
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
        data = response.json()
        return data["uuid"]

    def check_generation(self, request_id, attempts=10, delay=10):
        while attempts > 0:
            response = requests.get(
                self.URL + "key/api/v1/pipeline/status/" + request_id,
                headers=self.AUTH_HEADERS,
            )
            data = response.json()
            if data["status"] == "DONE":
                return data["result"]["files"]

            attempts -= 1
            time.sleep(delay)


if __name__ == "__main__":

    api_key = os.getenv("FUSIONBRAIN_API_KEY")
    secret_key = os.getenv("FUSIONBRAIN_SECRET_KEY")
    api = FusionBrainAPI("https://api-key.fusionbrain.ai/", api_key, secret_key)
    pipeline_id = api.get_pipeline()
    uuid = api.generate("Sun in sky", pipeline_id)
    files = api.check_generation(uuid)
    print(files)
