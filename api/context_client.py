import requests
import json
from llama_index.llms.openai import OpenAI

API_URL = "http://localhost:8000"  # Adjust if running in a different environment

def get_context():
    """Fetch the shared context from the API."""
    response = requests.get(f"{API_URL}/context/")
    if response.status_code != 200:
        raise ValueError(f"Failed to get context: {response.json()['detail']}")
    print("RESPONSE", type(response.json()), response.json())
    return response.json()["data"]

def set_context(ctx):
    """Update the shared context via the API."""
    # Serialize non-serializable objects
    def serialize(obj):
        if isinstance(obj, OpenAI):
            return {"type": "OpenAI", "model": obj.model, "temperature": obj.temperature}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    serialized_ctx = json.dumps(ctx, default=serialize)
    response = requests.post(f"{API_URL}/context/", json={"data": json.loads(serialized_ctx)})
    if response.status_code != 200:
        raise ValueError(f"Failed to set context: {response.json()['detail']}")