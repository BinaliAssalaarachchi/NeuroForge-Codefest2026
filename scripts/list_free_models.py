import requests
import json

def get_free_models():
    resp = requests.get("https://openrouter.ai/api/v1/models")
    if resp.status_code == 200:
        models = resp.json().get("data", [])
        free_models = [m["id"] for m in models if m.get("id", "").endswith(":free") or "free" in m.get("id", "")]
        print(f"Found {len(free_models)} free models on OpenRouter:")
        for fm in sorted(free_models):
            print(f" - {fm}")
    else:
        print("Failed to fetch models list:", resp.status_code, resp.text)

if __name__ == "__main__":
    get_free_models()
