import requests
from giza import API_HOST

MODEL_ID = None  # Update with your model ID
VERSION_ID = None  # Update with your version ID
DEPLOYMENT_ID = None  # Update with your deployment id
REQUEST_ID = None  # Update with your request id
API_KEY = None  # Update with your API key

url = f"{API_HOST}/api/v1/models/{MODEL_ID}/versions/{VERSION_ID}/deployments/{DEPLOYMENT_ID}/proofs/{REQUEST_ID}:download"

headers = {"X-API-KEY": API_KEY}

d_url = requests.get(url, headers=headers).json()["download_url"]

proof = requests.get(d_url)

with open("zk.proof", "wb") as f:
    f.write(proof.content)
