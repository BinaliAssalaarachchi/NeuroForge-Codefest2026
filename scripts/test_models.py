import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from src.utils.client_openrouter import OpenRouterClient

models_to_test = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free"
]

def main():
    for m in models_to_test:
        print(f"\n--- Testing Model: {m} ---")
        try:
            client = OpenRouterClient(model=m)
            response = client.generate(
                [{"role": "user", "content": 'Reply with JSON: {"status": "ok", "answer": "working"}'}],
                json_mode=True,
                use_cache=False
            )
            print(f"✅ SUCCESS with {m}!")
            print(f"Response: {response}")
            return m
        except Exception as e:
            print(f"❌ FAILED with {m}: {e}")
    return None

if __name__ == "__main__":
    main()
