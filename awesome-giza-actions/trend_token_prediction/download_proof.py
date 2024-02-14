from typing import Dict

import requests
from giza import API_HOST

MODEL_ID = ...  # Update with your model ID
VERSION_ID = ...  # Update with your version ID
DEPLOYMENT_ID = ...  # Update with your deployment id
REQUEST_ID = "..."  # Update with your request id
API_KEY = "..."  # Update with your API key, available at ~/.giza/

url = (
    f"{API_HOST}/api/v1/"
    "models/{MODEL_ID}/"
    "versions/{VERSION_ID}/"
    "deployments/{DEPLOYMENT_ID}/"
    "proofs/{REQUEST_ID}:download"
)

headers = {"X-API-KEY": API_KEY}


def download_proof(url: str, headers: Dict[str, str], file_path: str):
    """
    Download a deployment proof from giza

    Args:
        url (str): API url
        headers (Dict[str, str]): Auth headers
        file_path (str): Path to save the proof
    """
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    print(f"Saving proof to {file_path}")
    with open(file_path, "wb") as f:
        f.write(response.content)
    print("Proof saved successfully")
